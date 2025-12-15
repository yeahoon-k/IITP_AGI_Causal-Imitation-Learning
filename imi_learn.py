import multiprocessing
import random
import sys
from collections import defaultdict
from typing import AbstractSet, Dict, Tuple, Sequence, Collection

import networkx as nx
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from joblib import Parallel, delayed
from networkx import NetworkXUnfeasible

from npsem.model import CausalDiagram as CD, SCM
from npsem.model_utils import random_scm
from npsem.utils import pairs
from scil import seq_imi_graphs, seq_pi_bd, Policy

ASet = AbstractSet[str]

def mark(m):
    if m == -1:
        return '-', '-'
    if m == 1:
        return '<', '>'
    if m == 2:
        return 'o', 'o'


def causallearn_print(G, order):
    for i, j in pairs(list(range(len(order)))):
        if G[i, j] != 0:
            print(f'{order[i]} {mark(G[i, j])[0]}-{mark(G[j, i])[1]} {order[j]}')


# discrete conditional probability table
def cpt(data, X, Zs: Tuple, index: Dict[str, int], dom_X, epsilon=0.01) -> Dict[Tuple, Dict[str, float]]:
    loc_Zs = [index[Z] for Z in Zs]
    loc_X = index[X]

    # counts first
    cnts = defaultdict(lambda: {x: epsilon for x in dom_X})
    for x, zs in zip(data[:, loc_X], data[:, loc_Zs]):
        cnts[tuple(zs)][x] += 1.0

    # normalize
    cpts = defaultdict(lambda: {x: 1. / len(dom_X) for x in dom_X})
    for zs, x_cnts in list(cnts.items()):
        n_zs = sum(x_cnts.values())
        cpts[zs] = {x: x_cnts[x] / n_zs for x in dom_X}

    return cpts


# get_policy_funcs : 받은 policy를 사용함
def get_policy_funcs(data, index, policy: Policy, dom_Xs: Tuple[str, Sequence]):
    return {X: cpt_to_func(X, Zs, data, dom_Xs, index) for X, Zs in policy.items()}


def cpt_to_func(X, Zs, data, dom_Xs, index):
    x_given_zs = cpt(data, X, sorted(Zs), index, dom_Xs[X])

    def pi_x_given_zs(values):
        zs = tuple(values[Z] for Z in Zs)

        x_probs_dict = x_given_zs[zs]

        xs = list(x_probs_dict.keys())
        pxs = list(x_probs_dict.values())

        return xs[np.random.choice(len(pxs), 1, p=pxs)[0]]

    return pi_x_given_zs


def create_G_pi(ori_G: CD, policy: Policy):
    Xs = policy.keys()
    G = ori_G.do(Xs)
    newG = G + [(Z, X) for X, Zs in policy.items() for Z in Zs]
    try:
        newG.causal_order()
    except NetworkXUnfeasible as e:
        return None
    return newG


def create_M_pi(model: SCM,
                pis,
                policy: Dict[str, Collection[str]],
                ):
    H = create_G_pi(model.G, policy)  # policy graph
    if H is None:
        return None

    F_pi = {V: (pis[V] if V in pis else F_V)  # updated functions
            for V, F_V in model.F.items()}
    # updated model
    return SCM(H,
               F_pi,
               model.P_U,
               model.D,
               model.more_U | model.G.U,
               stochastic=True)


def imitate(model: SCM,
            data: np.ndarray,
            index: Dict[str, int],
            policy: Dict[str, Collection[str]],
            dom_Xs: Dict[str, Sequence], size=1000):

    pis = get_policy_funcs(data, index, policy, dom_Xs)

    Mpi = create_M_pi(model, pis, policy)
    if Mpi is not None:
        return Mpi.stochastic_sample(size)
    else:
        return None


def pc2cd(pc_adj, M: SCM):
    column_order = M.G.causal_order()

    N = len(column_order)
    dir_edges = []
    undir_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            #
            if pc_adj[i, j] == 1 and pc_adj[j, i] == -1:
                dir_edges.append((j, i))
            elif pc_adj[i, j] == -1 and pc_adj[j, i] == 1:
                dir_edges.append((i, j))
            elif pc_adj[i, j] == 1 and pc_adj[j, i] == 1:
                undir_edges.append((i, j))
            elif pc_adj[i, j] == -1 and pc_adj[j, i] == -1:
                undir_edges.append((i, j))

    # random permutation
    if not nx.is_directed_acyclic_graph(nx.DiGraph(dir_edges)):
        assert False

    perm = list(range(0, N))
    while True:
        random.shuffle(perm)
        if not all(perm.index(a) < perm.index(b) for a, b in dir_edges):
            continue
        newly_directed_edges = [(a, b) if perm.index(a) < perm.index(b) else (b, a) for a, b in undir_edges]

        if nx.is_directed_acyclic_graph(nx.DiGraph(dir_edges + newly_directed_edges)):
            return CD(M.G.V,
                      [(column_order[x], column_order[y]) for x, y in dir_edges + newly_directed_edges]
                      )


def baseline(M: SCM, data, Xs, Y, size, order=None) -> float:
    pc_G = pc(data, indep_test='chisq', stable=False)
    pc_adj = pc_G.G.graph

    learned_G = pc2cd(pc_adj, M)

    # seq pi bd -> get a policy
    policy = seq_pi_bd(learned_G, Xs, Y, order)
    if not (Xs <= policy.keys()):
        policy = {X: learned_G.pa(X) for X in Xs}

    index = {V: i for i, V in enumerate(M.G.causal_order())}
    play = imitate(M, data, index, policy, {X: [0, 1] for X in Xs}, size)
    if play is not None:
        hat_EY = np.mean(play[:, M.G.causal_order().index(Y)])
        return hat_EY
    else:
        policy = {X: set() for X in Xs}
        index = {V: i for i, V in enumerate(M.G.causal_order())}
        play = imitate(M, data, index, policy, {X: [0, 1] for X in Xs}, size)
        hat_EY = np.mean(play[:, M.G.causal_order().index(Y)])
        return hat_EY


def proposed(M, data, Xs, Y, size, order=None) -> float:
    # causal discovery
    # fci_G, fci_edges = fci(data, 'chisq')
    # fci_adj = fci_G.graph
    pc_G = pc(data, indep_test='chisq', stable=True, uc_rule=2, uc_priority=3)
    pc_adj = pc_G.G.graph

    learned_G = pc2cd(pc_adj, M)

    policy = seq_pi_bd(learned_G, Xs, Y, order)
    if not (Xs <= policy.keys()):
        policy = {X: learned_G.pa(X) for X in Xs}

    index = {V: i for i, V in enumerate(M.G.causal_order())}
    play = imitate(M, data, index, policy, {X: [0, 1] for X in Xs}, size)
    if play is not None:
        hat_EY = np.mean(play[:, M.G.causal_order().index(Y)])
        return hat_EY
    else:
        policy = {X: set() for X in Xs}
        index = {V: i for i, V in enumerate(M.G.causal_order())}
        play = imitate(M, data, index, policy, {X: [0, 1] for X in Xs}, size)
        hat_EY = np.mean(play[:, M.G.causal_order().index(Y)])
        return hat_EY


def main(n_repeats=10, sample_size=5000, n_cpu=5):
    # Table 1.
    MSE_bases, MSE_props = [], []

    for name in seq_imi_graphs().keys():
        aa, bb = _inner_(name, n_repeats, sample_size, n_cpu)
        MSE_bases.extend(aa)
        MSE_props.extend(bb)

    MSE_bases = np.array(MSE_bases)
    MSE_props = np.array(MSE_props)
    print(np.mean(MSE_bases), np.std(MSE_bases))
    print(np.mean(MSE_props), np.std(MSE_props))

    base_mean = np.mean(MSE_bases)
    base_std = np.std(MSE_bases)
    prop_mean = np.mean(MSE_props)
    prop_std = np.std(MSE_props)

    # diff
    abs_diff = base_mean - prop_mean
    rel_diff = abs_diff / base_mean * 100
    print(f"Rel diff: {rel_diff:.2f}%")


def _inner2_(name, sample_size, seed):
    np.random.seed(seed)
    MSE_bases, MSE_props = [], []
    G, Xs, Ys, order = seq_imi_graphs()[name]
    Y, *_ = Ys

    M = random_scm(G)
    data = M.sample(sample_size)
    true_EY = np.mean(data[:, G.causal_order().index(Y)])

    baseline_EY = baseline(M, data, Xs, Y, sample_size, order)
    proposed_EY = proposed(M, data, Xs, Y, sample_size, order)

    MSE_bases.append((baseline_EY - true_EY) ** 2)
    MSE_props.append((proposed_EY - true_EY) ** 2)

    return MSE_bases, MSE_props


def _inner_(name, n_repeats, sample_size, n_cpu):
    MSE_bases, MSE_props = [], []

    outputs = Parallel(n_jobs=n_cpu)(delayed(_inner2_)(name, sample_size, seed=_) for _ in range(n_repeats))
    print(outputs)
    for aa, bb in outputs:
        MSE_bases.extend(aa)
        MSE_props.extend(bb)

    return MSE_bases, MSE_props


if __name__ == '__main__':
    # Original graph
    np.random.seed(0)
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))




