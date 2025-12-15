from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Dict, Any, Generator, Tuple, Set, Sequence, Collection, AbstractSet, FrozenSet, List

import networkx as nx
import numpy as np

from npsem.idty.prob_eq import EqNode
from npsem.model import CD, StructuralCausalModel, StructuralCausalModel as SCM
from npsem.model_utils import default_P_U, print_prob_result, random_scm
from npsem.utils import nonempties, as_sets, combinations, xors, sortup, dict_ors, max_default, fzset_union, shuffled, dict_or

ASet = AbstractSet[str]
VSet = FrozenSet[str]


def is_c_forest(F: CD, Rs: ASet = None) -> bool:
    """ Whether the given graph F's is Rs-rooted c-forest """
    if not Rs:
        Rs = F.sinks()

    assert F.V and Rs <= F.V

    forestness = not F.ch(Rs) and max_default((len(F.ch(c)) for c in F.V - Rs), 1) == 1
    spanning_treeness = F.is_one_cc and len(F.U) == len(F.V) - 1
    return forestness and spanning_treeness


def is_super_c_forest(F: CD, Rs: ASet = None) -> bool:
    """ Whether the given graph F's edge-subgraph can be Rs-rooted c-forest """
    assert F.V

    if not Rs:
        # does not need to check ancestors of sinks are F.V
        return F.is_one_cc

    assert Rs <= F.V
    return F.An(Rs) == F.V and F.is_one_cc


def __trim_bidirected_edges(F: CD, F_: CD) -> Tuple[CD, CD]:
    """ Trim CVBE (connected via bidirected edges) to minimally CVBE. """
    nxcc = nx.connected_components

    xyus = list(F.biedges3)
    G = nx.Graph([(x0, y0) for x0, y0, _ in xyus])
    assert len(list(nxcc(G))) == 1
    G_ = G.subgraph(F_.V)
    assert len(list(nxcc(G_))) == 1

    while len(xyus) > len(F.V) - 1:
        for i in range(len(xyus)):
            G = nx.Graph([(x0, y0) for x0, y0, _ in xyus[:i] + xyus[i + 1:]])
            if set(G.nodes) != F.V or len(list(nxcc(G))) != 1:
                continue

            G_ = G.subgraph(F_.V)
            if set(G_.nodes) != F_.V or len(list(nxcc(G_))) != 1:
                continue

            xyus = xyus[:i] + xyus[i + 1:]
            break

    F = CD(set(), F.edges, xyus)
    F_ = F[F_.V & F.V]
    return F, F_


def __trim_directed_edges(F: CD, F_: CD, Rs: ASet) -> Tuple[CD, CD]:
    """ Trim directed edges while making sure both `F` and `F_` are C-forests rooted on `Rs` """
    F, F_ = F.underbar(Rs), F_.underbar(Rs)
    non_removable = set()

    for edge in set(F.edges) - non_removable:
        H = F - {edge}
        if H.An(Rs) != F.V or H[F_.V].An(Rs) != F_.V:
            non_removable.add(edge)
            continue
        F = H

    return F, F[F_.V]


def __make_it_forest_0(G: CD, Ys: ASet, Rs: ASet) -> CD:
    G = G.underbar(Ys)  # TODO <-- why?
    assert G.An(Ys) >= Rs
    non_removable = set()

    for edge in set(G.edges) - non_removable:
        H = G - {edge}
        if not Rs <= H.An(Ys):
            non_removable.add(edge)
            continue
        G = H

    G = G[G.An(Ys)]
    return G


def __trim_edges_to_hedge(F: CD, F_: CD, Rs: ASet) -> Tuple[CD, CD]:
    F, F_ = __trim_bidirected_edges(F, F_)
    F, F_ = __trim_directed_edges(F, F_, Rs)
    return F, F_


def __is_super_c_forest(F: CD, Rs: ASet) -> bool:
    if F.V == F.An(Rs):
        ccs = list(F[F.An(Rs)].c_components)
        return len(ccs) == 1 and ccs[0] == F.V

    return False


def demonstrate_non_idness(G: CD, Ys: ASet, Xs: ASet, Zs: ASet = frozenset(), verbose=False):
    """ Create two models for (z-)ID demonstrating non-z-identifiability of P(Ys|do(Xs)) in G """
    for F, F_, Rs in __enumerate_super_hedges(G, Ys, Xs):
        if not (F.V - F_.V).isdisjoint(Zs):
            continue

        # hedge part, non unique...
        F, F_ = __trim_edges_to_hedge(F, F_, Rs)

        # attaching part
        G_ = G[(G - Xs).De(Rs)]
        G_ = CD(G_.V, G_.edges)
        Ys_ = None
        for Ys_ in nonempties(as_sets(combinations(Ys & G_.V))):
            if Rs <= G_.An(Ys_):
                break
        assert Ys_
        G_ = G_.underbar(Ys_)
        G_ = __make_it_forest_0(G_, Ys_, Rs)
        H = G_ + F  # type: CD

        bits = {v: 1 for v in F.V}
        bits.update({v: 2 for v in G_.V})
        # actual bits
        outsiders = H.V - F.V
        more_u = {f'U_{outsider}' for outsider in outsiders}
        mu1 = {u: 0.5 for u in F.U | more_u}
        domains = defaultdict(lambda: (0, 1))  # type: Dict[str, Tuple[int,...]]
        for v in H.V:
            if bits[v] == 2:
                if v in outsiders:
                    domains[v] = (0, 2)
                else:
                    domains[v] = (0, 1, 2, 3)

        def funcxor1(vv):
            # noinspection DuplicatedCode
            def inner(vals):
                values = list()
                # 1st bits
                if vv in F.V:
                    for p in F.pa(vv) | F.UCs(vv):
                        values.append(vals[p] & 1)
                first_bit = xors(values)

                # 2nd bits
                values = list()
                if bits[vv] == 2:
                    if vv in Rs:
                        values.append(first_bit << 1)
                    for a, b in G_.edges:
                        if b == vv:
                            values.append(vals[a] & 2)
                second_bit = xors(values)
                curr_val = first_bit + second_bit

                # extra randomness
                if vv in outsiders:
                    return curr_val if vals[f'U_{vv}'] else 2
                return curr_val

            return inner

        def funcxor2(vv):
            # noinspection DuplicatedCode
            def inner(vals):

                values = list()
                # 1st bits
                if vv in F.V:
                    if vv in F_.V:
                        for p in F_.pa(vv) | F_.UCs(vv):
                            values.append(vals[p] & 1)
                    else:
                        for p in F.pa(vv) | F.UCs(vv):
                            values.append(vals[p] & 1)
                first_bit = xors(values)

                # 2nd bits
                values = list()
                if bits[vv] == 2:
                    if vv in Rs:
                        values.append(first_bit << 1)
                    for a, b in G_.edges:
                        if b == vv:
                            values.append(vals[a] & 2)
                second_bit = xors(values)
                curr_val = first_bit + second_bit

                # extra randomness
                if vv in outsiders:
                    return curr_val if vals[f'U_{vv}'] else 2
                return curr_val

            return inner

        Xs_ = Xs & H.V
        M1 = StructuralCausalModel(H,
                                   F={vv: funcxor1(vv) for vv in H.V},
                                   P_U=default_P_U(mu1),
                                   D=domains,
                                   more_U=more_u)
        M2 = StructuralCausalModel(H,
                                   F={vv: funcxor2(vv) for vv in H.V},
                                   P_U=default_P_U(mu1),
                                   D=domains,
                                   more_U=more_u)

        p1obs = M1.query(sortup(H.V))
        p2obs = M2.query(sortup(H.V))

        if verbose:
            print_prob_result([p1obs, p2obs], sortup(H.V), domains=domains)
        assert all(np.allclose(p1obs[k], p2obs[k]) for k in p1obs), 'diff obs'

        assert p1obs.keys() == p2obs.keys(), 'diff keys '
        x_val = {X: np.random.randint(2) for X in Xs_}
        p1do = M1.query(sortup(Ys_), intervention=x_val)
        p2do = M2.query(sortup(Ys_), intervention=x_val)
        if verbose:
            print_prob_result([p1do, p2do], sortup(Ys_), domains=domains)
        assert any(not np.allclose(p1do[k], p2do[k]) for k in set(p1do) | set(p2do)), 'same do'
        return


def validate_equation(G: CD,
                      Ys: ASet,
                      Xs: ASet,
                      equation: EqNode,
                      pass_how_many=1,
                      max_trial=20,
                      pass_for_max=False,
                      user_M=None,
                      Zs: ASet = frozenset(),
                      *,
                      allclose_kwargs: Optional[Dict[str, Any]] = None
                      ):
    """ Empirically evaluate the returned equation for P(Ys | do(Xs), Zs) with a randomly generated SCM, throws an exception if failed """
    if allclose_kwargs is None:
        allclose_kwargs = {}
    assert 0 < pass_how_many < max_trial
    domains = defaultdict(lambda: (0, 1))

    if user_M is not None:
        pass_how_many = 1
        max_trial = 1
        pass_for_max = False

    total_passed = 0
    # noinspection DuplicatedCode
    while max_trial:
        max_trial -= 1

        M = random_scm(G) if user_M is None else user_M
        xval = {x: np.random.randint(2) for x in sortup(Xs)}
        yval = {y: np.random.randint(2) for y in sortup(Ys)}
        zval = {z: np.random.randint(2) for z in sortup(Zs)}

        truth = M.query(sortup(Ys), intervention=xval, condition=zval)[tuple(yval.values())]
        equation.attach_distribution(M)
        estimated = equation.evaluate(dict_ors(xval, yval, zval), domains, zero_for_nan=True)
        if np.isnan(truth) or np.isnan(estimated):
            continue
        if estimated == 0.0 and truth != 0.0:
            continue
        if not np.allclose(truth, estimated, **allclose_kwargs):
            raise AssertionError(f'{abs(truth - estimated)}')
        total_passed += 1
        if total_passed >= pass_how_many:
            return
    if pass_for_max:
        return
    raise AssertionError(f"equation is invalid: {G} {Ys} {Xs} {Zs} {equation}")


def __enumerate_super_hedges(G: CD, Ys: ASet, Xs: ASet) -> Generator[Tuple[CD, CD, Set[str]], None, None]:
    """ Find a hedge-containing subgraph of G whose edge-subgraph corresponds to a hedge

    Not an efficient procedure
    """
    G = G[G.An(Ys)]
    Xs &= G.V
    Rs_able = G.do(Xs).An(Ys) - Xs
    for CCi in G.c_components:
        for Rs in nonempties(as_sets(combinations(CCi & Rs_able))):  # since Rs <= F' which should be CVBE
            H = G[G.An(Rs)]
            for sub_Fs in nonempties(as_sets(combinations(H.V - Rs))):  # We are checking sub_Fs such that sub_Fs = F \ F'
                if Xs.isdisjoint(sub_Fs):
                    continue

                if not __is_super_c_forest(F := H[sub_Fs | Rs], Rs):
                    continue

                for sub_Fs__ in as_sets(combinations(F.V - Rs - Xs)):  # check inner part
                    if __is_super_c_forest(F__ := H[sub_Fs__ | Rs], Rs):
                        yield F, F__, Rs


class NotIdentifiableError(BaseException):
    def __init__(self, msg, evidence=None):
        super().__init__(msg)
        self.evidence = evidence


@dataclass
class IDSetting:
    G: CD
    Ys: VSet
    Xs: VSet


@dataclass
class IDCSetting(IDSetting):
    Ws: VSet


@dataclass
class ZIDSetting(IDSetting):
    Zs: VSet


@dataclass
class GIDSetting(IDSetting):
    Zss: Collection[VSet]


@dataclass
class TRSetting(IDSetting):
    Ss: AbstractSet = field(default_factory=set)


class GTRCSetting(IDCSetting):
    Zss: Sequence[VSet]
    Sss: Sequence[VSet]


# noinspection NonAsciiCharacters
def compare_distributions_tr(M1: SCM,
                             M2: SCM,
                             Zss: Sequence[ASet],
                             Δ: Sequence[ASet],
                             outcome=None,
                             condition_vars=None,
                             intervention_vars=None,
                             condition_value=None,
                             intervention_value=None,
                             sampling=False):
    if condition_value is None:
        condition_value = dict()

    if intervention_value is None:
        intervention_value = dict()

    if condition_vars is not None:
        condition_values = [dict(zip(sortup(condition_vars), cs)) for cs in
                            product(*[M1.D[V] for V in sortup(condition_vars)])]
    else:
        assert condition_value is not None
        condition_values = [condition_value]

    if intervention_vars is not None:
        intervention_values = [dict(zip(sortup(intervention_vars), cs)) for cs in
                               product(*[M1.D[V] for V in sortup(intervention_vars)])]
    else:
        assert intervention_value is not None
        intervention_values = [intervention_value]

    G = M1.G
    Vs = sortup(G.V)

    all_diffs = fzset_union(Δ)
    for i, (Zss_i, Δ_i) in enumerate(zip(Zss, Δ)):
        not_Δ_i = all_diffs - Δ_i
        for Zs in Zss_i:
            zss = list(product(*[M1.D[Z] for Z in sortup(Zs)]))
            if sampling:
                zss = [shuffled(zss)[0]]
            for zs in zss:
                Zz = dict(zip(Zs, zs))
                Zz.update({f'S_{V}': i for V in Δ_i})
                Zz.update({f'S_{V}': 0 for V in not_Δ_i})
                P1 = M1.query(Vs, intervention=Zz)
                P2 = M2.query(Vs, intervention=Zz)

                assert all(np.allclose(P1[k], P2[k]) for k in
                           P1.keys() | P2.keys()), f"should be equal Vs:{Vs}, Zz:{Zz}"

    do_target_domain = {f'S_{V}': 0 for V in all_diffs}
    once = True

    for cv, iv in product(condition_values, intervention_values):
        if once:
            given_dict = dict_or(cv, iv)
            given_vars = sortup(given_dict.keys())
            P1 = M1.query(given_vars, intervention=do_target_domain)
            P2 = M2.query(given_vars, intervention=do_target_domain)
            assert P1[tuple(given_dict[V] for V in given_vars)] > 0
            assert P2[tuple(given_dict[V] for V in given_vars)] > 0
            # print('checked')
            once = False

        iv = dict_or(iv, do_target_domain)
        P1 = M1.query(outcome, cv, iv)
        P2 = M2.query(outcome, cv, iv)
        if not all(np.allclose(P1[k], P2[k]) for k in P1.keys() | P2.keys()):
            break
    else:
        raise AssertionError('should be different')


def compare_distributions(M1: SCM,
                          M2: SCM,
                          outcome: Collection[str] = None,
                          condition_vars: Collection[str] = None,
                          intervention_vars: Collection[str] = None,
                          condition_value: Dict[str, Any] = None,
                          intervention_value: Dict[str, Any] = None,
                          Zss: Collection[AbstractSet[str]] = tuple(),
                          sampling=False,
                          check_positivity=False,
                          verbose=False,
                          verbose_notable=False,
                          fast=False):
    """ For gID, check whether two models differ on the given ... while agreeing on all values


    If *vars is defined, all of its values will be tested (for condition and intervention)
    Else if *value is defined,

    If sampling is True, check only one value for given experimental distribution.

    """
    outcome = sortup(outcome)
    # condition
    if condition_value is None:
        condition_value = dict()

    if condition_vars is not None:
        condition_values = [dict(zip(sortup(condition_vars), cs)) for cs in
                            product(*[M1.D[V] for V in sortup(condition_vars)])]
    else:
        assert condition_value is not None
        condition_values = [condition_value]

    # intervention
    if intervention_value is None:
        intervention_value = dict()

    if intervention_vars is not None:
        intervention_values = [dict(zip(sortup(intervention_vars), cs)) for cs in
                               product(*[M1.D[V] for V in sortup(intervention_vars)])]
    else:
        assert intervention_value is not None
        intervention_values = [intervention_value]

    G = M1.G
    # Vs = sortup(G.V)

    for Zs in Zss:
        zss = list(product(*[M1.D[Z] for Z in sortup(Zs)]))
        if sampling:
            zss = [shuffled(zss)[0]]

        queried = sortup(G.V - set(Zs))
        for zs in zss:
            P1 = M1.query(queried, intervention=(Zz := dict(zip(Zs, zs))), fast=fast)
            P2 = M2.query(queried, intervention=Zz, fast=fast)

            to_pass = all(np.allclose(P1[k], P2[k]) for k in P1.keys() | P2.keys())
            if verbose and (not verbose_notable or not to_pass):
                print_prob_result([P1, P2], queried, {_: M1.D[_] for _ in queried})
            assert to_pass, f"should be equal queried:{queried}, Zz:{Zz}"
            if check_positivity:
                for k in M1.values_of(queried):
                    assert 0 < P1[k]
                    assert 0 < P2[k]

    for cv, iv in product(condition_values, intervention_values):
        P1 = M1.query(outcome, cv, iv, fast=fast)
        P2 = M2.query(outcome, cv, iv, fast=fast)

        to_pass = not all(np.allclose(P1[k], P2[k]) for k in P1.keys() | P2.keys())
        if verbose and (not verbose_notable or to_pass):
            print(outcome, cv, iv)
            print_prob_result([P1, P2], outcome, {_: M1.D[_] for _ in outcome})

        if to_pass:
            break
    else:
        raise AssertionError(f'should be different {outcome=}, {condition_vars=}, {intervention_vars=}')


# noinspection NonAsciiCharacters
def random_gtr_setting(G: CD) -> Tuple[List[List[FrozenSet[str]]], List[Set[str]]]:
    n_domains = min(max(2, np.random.poisson(2)), 4)
    Δ = {}
    Zss = {}
    for i in range(n_domains):
        if i == 0:
            Δ[i] = set()
        else:
            Δ[i] = set(np.random.choice(list(G.V), size=min(len(G.V), max(1, np.random.poisson(1.5)))))

        n_Zss_i = np.random.poisson(1.25)
        Zss_i = set()
        for _ in range(n_Zss_i):
            Zs = frozenset(V for V in G.V if np.random.rand() < 0.25)
            Zss_i.add(Zs)

        Zss[i] = list(Zss_i)
    return [Zss[i] for i in range(n_domains)], [Δ[i] for i in range(n_domains)]
