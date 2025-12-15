import functools
import warnings
from typing import Collection, Optional, List, Tuple, Set, FrozenSet, AbstractSet, Union

from npsem.idty.idty_utils import NotIdentifiableError
from npsem.idty.prob_eq import assign, Pr, ProbDistr, Σ, Π, EqNode, frac_node, ONE1
from npsem.idty.zid import Hedge, zID
from npsem.model import CD
from npsem.model_utils import trim_bidirected_edges_while
from npsem.stat_utils import ProbSpec
from npsem.utils import optional_next, shuffled, as_fzsets, summation

"""
Implements g-identifiability and its extensions

References
----------
Sanghack Lee, Juan D. Correa, and Elias Bareinboim, 
General Identifiability with Arbitrary Surrogate Experiments 
In Proceedings of the Thirty-fifth Conference on Uncertainty in Artificial Intelligence (UAI 2019)
"""

VSet = FrozenSet[str]
ASet = AbstractSet[str]

def CC(G: CD):
    """ c-components """
    return G.c_components

def essential_conditions(G: CD, Ys: Union[ProbSpec, ASet], Xs: ASet, Ws: ASet) -> Tuple[VSet, VSet]:
    """ Take out observation as many as possible using Rule 2 of do-calculus """
    if isinstance(Ys, ProbSpec):
        return essential_conditions(G, set(Ys.outcome), Ys.Xs, Ys.Zs)

    for W in shuffled(sorted(Ws)):  # reproducibility
        if (G - Xs).underbar(W).independent(Ys, {W}, Ws - {W}):
            return essential_conditions(G, Ys, Xs | {W}, Ws - {W})
    return frozenset(Ws), frozenset(Xs)


def is_gIDC(G: CD, Ys: ASet, Xs: ASet, Ws: ASet, Zss: Collection[ASet]) -> bool:
    """ see `gIDC` """
    if isinstance(Ys, str):
        Ys = {Ys}
    if isinstance(Xs, str):
        Xs = {Xs}
    if not Ys:
        return True

    Ws, Xs = essential_conditions(G, Ys, Xs, Ws)

    return is_gID(G, Ys | Ws, Xs, Zss)


def gIDC(G: CD, Ys: ASet, Xs: ASet, Ws: ASet, Zss: Collection[ASet]) -> EqNode:
    """
    An extension of `gID` for conditional interventional distribution queries

    Sanghack Lee, Juan D. Correa, and Elias Bareinboim,
    General Identifiability with Arbitrary Surrogate Experiments.
    In Proceedings of the Thirty-fifth Conference on Uncertainty in Artificial Intelligence (UAI 2019)
    """
    if isinstance(Ys, str):
        Ys = {Ys}
    if isinstance(Xs, str):
        Xs = {Xs}
    if not Ys:
        return ONE1

    Ws, Xs = essential_conditions(G, Ys, Xs, Ws)

    return frac_node(f := gID(G, Ys | Ws, Xs, Zss),
                     Σ(Ys, f))


def is_gID(G: CD, Ys: ASet, Xs: ASet, Zss: Collection[ASet]) -> bool:
    """ see `gID` """
    if isinstance(Ys, str):
        Ys = {Ys}
    if isinstance(Xs, str):
        Xs = {Xs}
    assert Xs.isdisjoint(Ys)
    if not Ys:
        return True

    Vs = G.V
    Zs = optional_next(filter(lambda _Zs: Xs == _Zs & Vs, Zss))
    if Zs is not None:
        return True

    if Vs != G.An(Ys):
        return is_gID(G[G.An(Ys)], Ys, Xs & G.An(Ys), Zss)

    if Ws := (Vs - Xs) - G.do(Xs).An(Ys):
        return is_gID(G, Ys, Xs | Ws, Zss)

    if len(CCs := CC(G - Xs)) > 1:
        return all(is_gID(G, S_i, Vs - S_i, Zss) for S_i in sorted(CCs))

    return any(is_subID(G - (Zs & Vs), Ys, Xs - Zs) for Zs in filter(lambda _Zs: _Zs & Vs <= Xs, Zss))

def gID(G: CD, Ys: Union[ASet, str], Xs: Union[ASet, str], Zss: Collection[ASet], *, no_thicket=False) -> EqNode:
    """ Formula for P(Ys|do(Xs)) given experiments `Zss` in G

    References
    ----------
    Sanghack Lee, Juan D. Correa, and Elias Bareinboim,
    General Identifiability with Arbitrary Surrogate Experiments.
    In Proceedings of the Thirty-fifth Conference on Uncertainty in Artificial Intelligence (UAI 2019)
    """

    if isinstance(Ys, str):
        Ys = {Ys}
    if isinstance(Xs, str):
        Xs = {Xs}
    assert Xs.isdisjoint(Ys)
    if not Ys:
        return ONE1

    Vs = G.V
    Zs = optional_next(filter(lambda _Zs: Xs == _Zs & Vs, Zss))
    if Zs is not None:
        return assign(Pr(ProbDistr(Zs, G.V), Ys), Zs - Vs)
    print("---"*5)
    print('Vs:',Vs)
    print('An(Ys):', G.An(Ys))
    print("Ys:", Ys, "Xs:", Xs)
    # Y's anscestor가 Vs 아니면 : An(C) != T
    if Vs != G.An(Ys):
        print('if Vs != G.An(Ys):')
        return gID(G[G.An(Ys)], Ys, Xs & G.An(Ys), Zss, no_thicket=no_thicket)

    if Ws := (Vs - Xs) - G.do(Xs).An(Ys): # ... -> X -> .... -> Y
        print('if Ws := (Vs - Xs) - G.do(Xs).An(Ys):')
        return assign(gID(G, Ys, Xs | Ws, Zss, no_thicket=no_thicket), Ws)

    if len(CCs := CC(G - Xs)) > 1: # C-components # ... -> X -> .... -> Y
        print('if len(CCs := CC(G - Xs)) > 1:')
        print('CCs:',CCs)
        # test_sum = Σ(Vs - (Xs | Ys), Π([gID(G, S_i, Vs - S_i, Zss, no_thicket=no_thicket) for S_i in sorted(CCs)]))
        # print()
        # print("test_sum:",test_sum)
        return Σ(Vs - (Xs | Ys), Π([gID(G, S_i, Vs - S_i, Zss, no_thicket=no_thicket) for S_i in sorted(CCs)]))

    for Zs in filter(lambda _Zs: _Zs & Vs <= Xs, Zss):
        print('for Zs in filter(lambda _Zs: _Zs & Vs <= Xs, Zss):')
        formula = assign(subID(G - (Zs & Vs), Ys, Xs - Zs, ProbDistr(Zs, Vs)), Zs - Vs)
        if formula is not None:
            print('formula return')
            return formula

    raise NotIdentifiableError('not identifiable', __find_thicket(G, Ys, Xs, Zss) if not no_thicket else None)

def subID(G: CD, Ys: ASet, Xs: ASet, P_I: EqNode) -> Optional[EqNode]:
    S, = CC(G - Xs)
    Vs = G.V

    if not Xs:
        return Σ(Vs - Ys, Pr(P_I, Vs))

    if Vs - G.An(Ys):
        return subID(G[G.An(Ys)], Ys, Xs & G.An(Ys), Σ(Vs - G.An(Ys), P_I))  # done

    # noinspection NonAsciiCharacters
    π = G.causal_order()

    def pre(qq):
        return frozenset(π[:π.index(qq)])

    if CC(G) == {Vs}:
        return None

    if S in CC(G):
        subformula = Π([Pr(P_I, {Q}, pre(Q), scope=Vs) for Q in sorted(S)])
        return Σ(S - Ys, subformula)

    S__ = optional_next(S__ for S__ in CC(G) if S < S__)
    P_I_ = Π([Pr(P_I, {Vi}, pre(Vi), scope=Vs) for Vi in sorted(S__)])
    return subID(G[S__], Ys, Xs & S__, P_I_)


def is_subID(G: CD, Ys: ASet, Xs: ASet) -> bool:
    S, = CC(G - Xs)
    Vs = G.V

    if not Xs:
        return True

    if Vs - G.An(Ys):
        return is_subID(G[G.An(Ys)], Ys, Xs & G.An(Ys))  # done

    if CC(G) == {Vs}:
        return False

    if S in CC(G):
        return True

    S__ = optional_next(S__ for S__ in CC(G) if S < S__)
    return is_subID(G[S__], Ys, Xs & S__)


def factors(G: CD, Ys: ASet, Xs: ASet = frozenset()) -> Tuple[CD, Set[ProbSpec]]:
    """ gc-factors """
    G = G[G.An(Ys)]
    Xs &= G.An(Ys)
    Xs |= (G.V - Xs) - G.do(Xs).An(Ys)  # Xs^+
    return G, {ProbSpec(G.pa(S_i) - S_i, S_i) for S_i in sorted(CC(G - Xs))}


def factor_eq(G: CD, Ys: ASet, Xs: ASet) -> EqNode:
    """  """
    G = G[G.An(Ys)]
    Xs &= G.An(Ys)
    Ws = (G.V - Xs) - G.do(Xs).An(Ys)
    Xs |= Ws

    prods = Π([Pr(ProbDistr(G.pa(S_i) - S_i, vs=G.V), S_i)
               for S_i in sorted(CC(G - Xs))])
    sum_prods = Σ(G.V - (Xs | Ys), prods)
    return assign(sum_prods, Ws)


def find_super_hedge(G: CD, Ys: ASet, Xs: ASet) -> Optional[Tuple[ASet, ASet]]:
    warnings.warn('dep', category=DeprecationWarning)
    S, = CC(G - Xs)
    Vs = G.V

    if not Xs:
        return None

    if Vs - G.An(Ys):
        return find_super_hedge(G[G.An(Ys)], Ys, Xs & G.An(Ys))  # done

    if CC(G) == {Vs}:
        return Ys, Xs

    if S in CC(G):
        return None

    S__ = optional_next(S__ for S__ in CC(G) if S < S__)
    return find_super_hedge(G[S__], Ys, Xs & S__)


class Thicket:
    def __init__(self, nondegenerate_hedges: Collection[Hedge], Rs_G: CD):
        if nondegenerate_hedges:
            self.hedges = tuple(nondegenerate_hedges)  # type: Tuple[Hedge]
        else:
            self.hedges = tuple([Hedge(Rs_G, Rs_G.V)])  # type: Tuple[Hedge]
        self.Rs = Rs_G.V
        self.aggregated = functools.lru_cache(maxsize=1)(self.aggregated)

    def aggregated(self) -> CD:
        return summation(hedge.F for hedge in self.hedges)  # type: CD

    def is_degenerate(self):
        return len(self.hedges) == 1 and self.hedges[0].F.V == self.hedges[0].F_.V

    def __str__(self):
        # return f'{self.hedges} on Rs={set(self.Rs)}'
        return f'{self.hedges}'

    def __repr__(self):
        if self.is_degenerate():
            return f'{self.__class__.__name__}([], {repr(self.hedges[0].F)})'
        else:
            return f'{self.__class__.__name__}({repr(self.hedges)}, {repr(self.hedges[0].F[self.Rs])})'


def hedgelets_of(hedge: Hedge, Rs: ASet) -> List[CD]:
    F = hedge.F
    hedge_top = (F - Rs).V
    if hedge_top:
        return [F[Rs | Ts] + F[F.De(Ts)].without_UCs() for Ts in sorted(F[hedge_top].c_components)]
    else:
        # degenerate
        return [hedge.F]


def __find_thicket(G: CD, Ys: ASet, Xs: ASet, Zss: Collection[ASet]) -> Thicket:
    assert Ys <= G.V
    assert Xs <= G.V
    assert (G - Xs).is_one_cc

    Ys, Xs, Zss = frozenset(Ys), frozenset(Xs), list({Zs & G.V for Zs in as_fzsets(Zss)})
    Vs = G.V

    hedges = set()
    for Zs in filter(lambda _Zs: _Zs & Vs <= Xs, Zss):
        try:
            zID(G - Zs, Ys, Xs - Zs)
            assert False, "can't be here"
        except NotIdentifiableError as e:
            hedges.add(e.evidence)

    assert len({h.Rs for h in hedges}) <= 1
    bidirected_only_over_Rs = trim_bidirected_edges_while(lambda G_: G_.is_one_cc, G - Xs).edges_removed((G - Xs).edges)
    return Thicket(hedges, bidirected_only_over_Rs)


def compute_pxy(G: CD, Ys: Union[ASet, str], Xs: Union[ASet, str]):
    CC_G = CC(G)
    S_X, S_k = 0, set()

    for e in CC_G:
        if e.intersection(Xs):
            S_X=e
        else:
            S_k.add(e)

    D = (G-Xs).An(Ys)
    D_X = D.intersection(S_X)
    CC_DX=G[D_X].c_components

    return all([PosID(Djx, S_X, G) for Djx in CC_DX])

def PosID(C :Union[ASet, str], T:Union[ASet, str], G: CD) -> bool:
    """
    An identify function determining if Q[C] is computable from Q[T] with positivity checking

    References
    ----------
    Jin Tian and Judea Pearl
    On the Identification of Causal Effects.
    """
    A = G[T].An(C)
    if A == C:
        return True
    if A == T:
        return False
    T_prime, T_i = 0, set()

    for e in G[A].c_components:
        if e.intersection(C):
            T_prime=e
        else:
            T_i.add(e)

    if not PosChecker(T_i, G):
        return False

    return PosID(C, T_prime, G)

def PosChecker(S, G):
    """
    Positivity function determining if the probability of c-factors are greater than zero
    """

    return True

if __name__ == "__main__":
    # Napkin Graph
    identifiability = compute_pxy(G := CD('W->R->X->Z->Y, X<->W<->Z'), {'Y'}, {'X'})
    # Bow Graph
    #identifiability = compute_pxy(G := CD('X->Y, X<->Y'), {'Y'}, {'X'})

    print('identifiability:', identifiability)
