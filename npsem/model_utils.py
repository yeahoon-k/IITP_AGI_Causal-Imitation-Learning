import itertools
from collections import defaultdict
from itertools import product
from typing import Sequence, Collection, AbstractSet, List, Optional, Tuple, Mapping, Callable, Dict, TypeVar, FrozenSet, Any, Union, Iterable

import numpy as np

from npsem.model import CD, qcd, StructuralCausalModel as SCM
from npsem.stat_utils import ProbSpec
from npsem.utils import shuffled, rand_bw, dict_or, fair_coin, seeded, missingdict

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')
VSet = FrozenSet[str]
ASet = AbstractSet[str]


def frontdoor_graph(outcome='Y', mediator='Z', intervention='X') -> CD:
    return qcd([[intervention, mediator, outcome]], [[intervention, outcome]])


def random_scm(G: CD) -> SCM:
    """ Randomly generated SCM with binary variables

    see `random_binary_func`
    """
    mu1 = {f'U_{v}': rand_bw(0.2, 0.8, precision=2) for v in sorted(G.V)}
    new_Us = {f'U_{v}' for v in sorted(G.V)}
    assert G.U.isdisjoint(new_Us)
    mu1 = dict_or(mu1, {u: rand_bw(0.2, 0.8, precision=2) for u in G.U})

    domains = defaultdict(lambda: (0, 1))  # type: Dict[str, Tuple[Any,...]]

    # SCM with parametrization
    return SCM(G,
               F={v: random_binary_func(G, v) for v in sorted(G.V)},
               P_U=default_P_U(mu1),
               D=domains,
               more_U={f'U_{v}' for v in G.V})


def random_transportable_scms(G: CD, Deltas: Sequence[Collection[ASet]]) -> List[SCM]:
    domains = defaultdict(lambda: (0, 1))

    more_U = {f'U_{v}' for v in G.V}
    assert G.U.isdisjoint(more_U)
    mu1 = {f'U_{v}': rand_bw(0.2, 0.8, precision=2) for v in sorted(G.V)}
    mu1 = dict_or(mu1, {u: rand_bw(0.2, 0.8, precision=2) for u in G.U})

    F1 = {v: random_binary_func(G, v) for v in sorted(G.V)}

    # SCM with parametrization
    M1 = SCM(G, F=F1, P_U=default_P_U(mu1), D=domains, more_U=more_U)

    Ms = [M1]
    for delta_i in Deltas[1:]:
        mu_i = dict(mu1)
        mu_i.update({f'U_{v}': rand_bw(0.2, 0.8, precision=2) for v in delta_i if fair_coin()})

        M = SCM(G,
                F={v: random_binary_func(G, v) if v in delta_i else F1[v] for v in sorted(G.V)},
                P_U=default_P_U(mu_i),
                D=domains,
                more_U=more_U)
        Ms.append(M)

    return Ms


def random_GYsXsZs(max_size: Optional[int] = 7,
                   n_U_func: Optional[Callable[[int], int]] = None,
                   seed: Optional[int] = None,
                   Y_rootset: bool = False,
                   n_V: Optional[int] = None,
                   n_edges: Optional[int] = None,
                   zx_disjoint: bool = False,
                   min_Xs_size: int = 0
                   ) -> Tuple[CD, VSet, VSet, VSet]:
    """ Randomly generate a causal graph with three subsets named Ys, Xs, Zs where Ys and Xs and Ys and Zs are disjoint.

    max_size: if n_V is not specified, it returns graphs of size [2, max_size]
    n_V: if specified, max_size is ignored, and a graph with n_V vertices is returned.
    seed: a random seed for reproducibility.
    zx_disjoint, if true, makes sure that Zs and Xs are disjoint.
    """
    with seeded(seed):
        if n_V is None:
            assert 2 <= max_size
            sizes = list(range(2, max_size + 1))
            pp = np.array(sizes)
            for i in range(1, len(sizes)):
                pp[i] *= pp[i - 1]
            pp = pp / np.sum(pp)

            n_V = np.random.choice(sizes, p=pp)

        if n_U_func is None:
            n_U = min(np.random.poisson(n_V - 1), n_V * (n_V - 1) // 2)
        else:
            n_U = n_U_func(n_V)

        if n_edges is None:
            n_edges = min(np.random.poisson(n_V - 1), n_V * (n_V - 1) // 2)
        G = random_causal_diagram(n_V, n_edges, n_U)
        if Y_rootset:
            Ys = G.root_set()
            nYs = len(Ys)
        else:
            nYs = max(1, min(len(G.V) - min_Xs_size, np.random.poisson(1.25)))  # -1 for Xs
            Ys = frozenset(np.random.choice(sorted(G.V), nYs, replace=False))

        nXs = max(min_Xs_size, min(len(G.V) - nYs, np.random.poisson(1.25)))
        nZs = min(len(G.V) - nYs - (nXs if zx_disjoint else 0), np.random.poisson(1.25))

        assert len(G.V - Ys) >= nXs
        Xs = frozenset(np.random.choice(sorted(G.V - Ys), nXs, replace=False))

        ZX = Xs if zx_disjoint else frozenset()
        if len(G.V - Ys - ZX) >= nZs > 0:
            Zs = frozenset(np.random.choice(sorted(G.V - Ys - ZX), nZs, replace=False))
        else:
            Zs = frozenset()
        return G, Ys, Xs, Zs


def perm_eq(g1: CD,
            g2: CD,
            renamer: Optional[Dict[str, str]] = None,
            permutable_pairs: Optional[Collection[Tuple[str, str]]] = None) -> bool:
    """ Whether two causal diagrams are topologically equivalent up to permutation of variable names.

    """
    if g1.characteristic == g2.characteristic:
        # TODO make use of a causal order
        g1_names = list(g1.V)
        g2_names = list(g2.V)
        for permed in itertools.permutations(g1_names):
            if permutable_pairs:  # TODO efficiency
                if not all((v1, v2) in permutable_pairs for v1, v2 in zip(permed, g2_names)):
                    continue

            if g1.renamed(dict(zip(permed, g2_names))) == g2:
                if renamer is not None:
                    renamer.update(dict(zip(permed, g2_names)))
                return True
    return False


def print_prob_result(ps: Sequence[Mapping[Any, float]],
                      variables: Tuple[str, ...],
                      domains: Optional[Mapping[str, Collection[Any]]] = None,
                      print_all=False):
    """ Print two probability distributions (as dicts) side by side """
    if domains is None:
        domains = defaultdict(lambda: (0, 1))
    space_for_vars = [len(str(v)) for v in variables]
    for i, var_name in enumerate(variables):
        for v in domains[var_name]:
            space_for_vars[i] = max(space_for_vars[i], len(str(v)))
    space_for_p = [max(len(str(v)) for v in p.values()) if p else 0 for p in ps]

    print('-' * (sum(space_for_vars) + len(space_for_vars) + 1 + sum(space_for_p) + 2 * len(space_for_p)))
    for i, v in enumerate(variables):
        print(f'{str(v).ljust(space_for_vars[i])} ', end='')

    print()
    for vals in product(*[domains[v] for v in variables]):
        if print_all or any(vals in p for p in ps):
            for i, val in enumerate(vals):
                print(f'{str(val).rjust(space_for_vars[i])} ', end='')
            print('|', end='')
            for i, p in enumerate(ps):
                print(f"  {str(p[vals]).rjust(space_for_p[i])}", end='')
            print()
    print('-' * (sum(space_for_vars) + len(space_for_vars) + 1 + sum(space_for_p) + 2 * len(space_for_p)))
    return


def print_distributions(M1: SCM, M2: SCM, test_probs: Union[ProbSpec, Iterable[ProbSpec]]):
    """ Print distributions for two models """
    for prob in [test_probs] if isinstance(test_probs, ProbSpec) else test_probs:
        xss = list(itertools.product(*[M1.D[X] for X in prob.X]))
        for xs in xss:
            Xx = dict(zip(prob.X, xs))
            print(prob, Xx)
            P1 = M1.query(prob.Y, intervention=Xx)
            P2 = M2.query(prob.Y, intervention=Xx)

            print_prob_result([P1, P2], prob.Y, M1.D)


def find_a_dpath_graph(G: CD, x: str, y: str, zs: ASet):
    # dumb, slow
    abus = {(a, b, None) for a, b in G.edges} | G.biedges3

    for a, b, u in shuffled(sorted(abus)):
        abu = (a, b) if u is None else (a, b, u)
        temp = G.edges_removed([abu])
        if not temp.independent(x, y, zs):
            G = temp

    for V in G.V:
        if not (G.pa(V) | G.ch(V) | G.UCs(V)):
            G = G - {V}

    return G


def trim_edges_while(func, G: CD, edges_to_keep=frozenset()) -> CD:
    return trim_directed_edges_while(func,
                                     trim_bidirected_edges_while(func, G, edges_to_keep),
                                     edges_to_keep)


def trim_directed_edges_while(func, G: CD, edges_to_keep=frozenset()) -> CD:
    """ Randomly remove directed edges if the resulting graph meets a given criterion.

    Every edge is tested once.
    """
    for a, b in shuffled(sorted(set(G.edges) - set(edges_to_keep))):
        temp = G.edges_removed([(a, b)])
        if func(temp):
            G = temp

    return G


def trim_bidirected_edges_while(func, G: CD, edges_to_keep=frozenset()) -> CD:
    """ Randomly remove UCs if the resulting graph meets a given criterion.

    Every UC is tested once.
    """
    for a, b, u in shuffled(sorted(G.biedges3 - set(edges_to_keep))):
        temp = G.edges_removed([(a, b, u)])
        if func(temp):
            G = temp
    return G


def random_causal_diagram(n_V=7, n_edges=10, n_U=10, seed=None, keep_alpha_order=True) -> CD:
    """ Returns a random causal diagram with specified number of vertices, bidirected edges, and directed edges.

    uses uppercase letters as vertex names
    keep_alpha_order: if set, the alphabetical order corresponds to a causal order. Otherwise, randomized.
    """
    alphas = list(reversed('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    with seeded(seed):
        if n_V <= len(alphas):
            names = alphas[:n_V]
        else:
            names = ['V' + str(i).zfill(int(np.ceil(np.log10(n_V)))) for i in range(n_V)]
        if not keep_alpha_order:
            np.random.shuffle(names)
        vs = set(names)
        pairs1, pairs2 = set(), set()
        for setvar, size in zip([pairs1, pairs2], [n_edges, n_U]):
            pairs = np.random.choice((n_V * (n_V - 1)) // 2, size, replace=False)
            offsets = np.cumsum(np.arange(n_V, 0, -1)) - n_V  #
            for edge_id in pairs:

                from_V = np.searchsorted(offsets, edge_id)

                if offsets[from_V] != edge_id:
                    from_V -= 1
                to_V = edge_id - offsets[from_V] + from_V + 1
                setvar.add((names[from_V], names[to_V]))

        return CD(vs, pairs1, {(x, y, 'U_' + str(i)) for i, (x, y) in enumerate(pairs2)})


def random_binary_func(G: CD, var: str) -> Callable[[Dict[str, int]], int]:
    """ Assuming U_{var} is valid variable specific disturbance ... """
    # make U_{var} explicitly xor of the evaluated
    # pas = G.pa(var) | {u for u, xy in G.confounded_dict.items() if var in xy} | {f'U_{var}'}
    pas = G.pa(var) | {u for u, xy in G.u2vv.items() if var in xy}
    if not pas:
        def inner2(v):
            return v[f'U_{var}']

        return inner2
    assert pas
    pas = sorted(pas)
    # 0: and 1: or 2: xor
    operators = list(np.random.randint(0, 3, len(pas) - 1))  # 0 and, 1 or , 2 xor
    postfix = pas + operators
    np.random.shuffle(postfix)
    while True:
        cnt_numbers = 0
        for i, cell in enumerate(postfix):
            if not isinstance(cell, str):
                if cnt_numbers < 2:
                    postfix.pop(i)
                    postfix.append(cell)
                    break
                else:
                    cnt_numbers -= 1
            else:
                cnt_numbers += 1
        else:
            break

    to_not = np.random.poisson(max(len(pas) // 2, 1))
    if to_not <= len(pas):
        insert_indices = np.random.choice(len(pas), to_not, replace=False)
        insert_indices = insert_indices[insert_indices > 0]
        insert_indices = sorted(insert_indices, reverse=True)
        for at in insert_indices:
            postfix.insert(at, 3)
    # vals = {'a': 0, 'b': 0, 'c': 1, 'd': 0, 'e': 0, 'f': 1, 'g': 0}
    operators = ['and', 'or', 'xor', 'not']

    def inner(v: Dict) -> int:
        array = [v[i_] if isinstance(i_, str) else operators[i_] for i_ in postfix]
        while len(array) != 1:
            for at_, cell_ in enumerate(array):
                if cell_ in operators:
                    if cell_ == 'not':
                        array[at_ - 1] = 1 - array[at_ - 1]
                        array.pop(at_)
                        break
                    elif cell_ == 'and':
                        array[at_ - 2] = array[at_ - 2] & array[at_ - 1]
                        array.pop(at_)
                        array.pop(at_ - 1)
                        break
                    elif cell_ == 'xor':
                        array[at_ - 2] = array[at_ - 2] ^ array[at_ - 1]
                        array.pop(at_)
                        array.pop(at_ - 1)
                        break
                    elif cell_ == 'or':
                        array[at_ - 2] = array[at_ - 2] | array[at_ - 1]
                        array.pop(at_)
                        array.pop(at_ - 1)
                        break
        return array[0] ^ v[f'U_{var}']

    return inner


def default_P_U(mu: Mapping[str, float], ignorable=True) -> Callable[[Dict[str, T]], float]:
    """ P(U) function given a dictionary of probabilities for each U_i being 1, P(U_i=1) """

    def P_U(d: Dict[str, T]) -> float:
        p_val = 1.0
        for k in mu.keys():
            if ignorable:
                if k in d:
                    p_val *= (1 - mu[k]) if d[k] == 0 else mu[k]
            else:
                p_val *= (1 - mu[k]) if d[k] == 0 else mu[k]
        return p_val

    return P_U


def to_tikzpicture(G: CD, coordinateless=False, renamer=None):
    if renamer is None:
        renamer = missingdict(lambda _: _)

    def subfy(s):
        for k in range(1, len(s)):
            if s[:k].isalpha() and s[k:].isnumeric():
                return s[:k] + '_' + s[k:]
        return s

    # _experimental,
    if coordinateless:
        ordered = G.causal_order()
        v = ordered[0]
        shift = -len(ordered) * 2
        shift_delta = len(ordered)
        print(r'\begin{figure}[H]')
        print(r'    \begin{tikzpicture}')
        print(rf'        \node ({renamer[v]}) {{${subfy(renamer[v])}$}};')
        for v, prev in zip(ordered[1:], ordered):
            print(
                rf'        \node[yshift={shift}mm] ({renamer[v]}) [right=of {renamer[prev]}] {{${subfy(renamer[v])}$}};')
            shift += shift_delta
            shift_delta -= 2
        for x, y in G.edges:
            print(rf'        \draw[->] ({renamer[x]}) -- ({renamer[y]});')
        for (x, y, u), direc in zip(G.biedges3, itertools.cycle(['left', 'right'])):
            print(rf'        \draw[<->,dashed] ({renamer[x]}) to [bend {direc}=45] ({renamer[y]});')
        print(r'    \end{tikzpicture}')
        print(r'\end{figure}')
    else:
        ordered = G.causal_order()
        v = ordered[0]
        dy = 0.
        dx = +2.
        coord_x = 0
        coord_y = 0
        print(r'\begin{figure}')
        print(r'    \begin{tikzpicture}')
        print(rf'        \node ({renamer[v]}) at ({coord_x:.3f}, {coord_y:.3f}) {{${subfy(renamer[v])}$}};')
        coord_x += dx
        coord_y += dy
        dx -= 0.4
        dy += 0.4
        for v, prev in zip(ordered[1:], ordered):
            print(rf'        \node ({renamer[v]}) at ({coord_x:.3f},{coord_y:.3f}) {{${subfy(renamer[v])}$}};')
            coord_x += dx
            coord_y += dy
            dx -= 0.4
            dy += 0.4
        for x, y in G.edges:
            print(rf'        \draw[->] ({renamer[x]}) -- ({renamer[y]});')
        for x, y, u in G.biedges3:
            print(rf'        \draw[<->,dashed] ({renamer[x]}) to [bend left=45] ({renamer[y]});')
        print(r'    \end{tikzpicture}')
        print(r'\end{figure}')


def random_causal_order(G: CD, among: VSet) -> List[str]:
    """ Return a valid causal order which depends on the current random seed """
    import networkx as nx
    try:
        DG = nx.DiGraph(G.edges)
        DG.add_nodes_from(among)  # if a variable does not have directed edge to any other variables...
    except nx.NetworkXError as e:
        raise AssertionError(str(G.edges)) from e

    sinks = [V for V in DG.nodes if not any(True for _ in DG.successors(V))]
    ordered = []  # first, bottom
    while sinks:
        sinks = shuffled(sinks)
        ordered.append(selected := sinks.pop())
        sinks += list(DG.predecessors(selected))
        sinks = set(sinks)
        DG.remove_node(selected)

    return list(reversed([v for v in ordered if v in among]))
