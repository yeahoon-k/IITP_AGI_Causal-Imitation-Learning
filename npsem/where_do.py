from typing import List, Tuple, FrozenSet, AbstractSet, Optional, Sequence

from npsem.model import CD
from npsem.utils import pop, only, combinations, with_default, as_fzsets

"""
References

Sanghack Lee and Elias Bareinboim (2018) 
Structural Causal Bandits: Where to Intervene?
Advances in Neural Information Processing Systems 31 (NeurIPS 2018)

Sanghack Lee and Elias Bareinboim (2019) 
On Structural Causal Bandit with Non-manipulable Variables.
In Proceedings of Thirty-third Conference on Artificial Intelligence (AAAI 2019) 
"""

VSet = FrozenSet[str]


def CC(G: CD, X: str) -> VSet:
    """ an X-containing c-component of G """
    return G.c_component(X)


def MISs(G: CD, Y: str, Ns: Optional[VSet] = None) -> FrozenSet[VSet]:
    """ Minimal Intervention Sets """
    Ns = with_default(Ns, frozenset())
    II = G.V - Ns - {Y}

    G = G[G.An(Y)]
    Ws = G.causal_order(backward=True)
    Ws = only(Ws, II)
    return subMISs(G, Y, frozenset(), Ws)


def subMISs(G: CD, Y: str, Xs: VSet, Ws: Sequence[str]) -> FrozenSet[VSet]:
    """ subroutine for MISs -- this creates a recursive call tree with n, n-1, n-2, ... widths """
    out = frozenset({Xs})
    for i, W_i in enumerate(Ws):
        H = G.do({W_i})
        H = H[H.An(Y)]
        out |= subMISs(H, Y, Xs | {W_i}, only(Ws[i + 1:], H.V))
    return out


def bruteforce_POMISs(G: CD, Y: str) -> FrozenSet[VSet]:
    """ This computes a complete set of POMISs in a brute-force way """
    return frozenset(frozenset(IB(G.do(Ws), Y))
                     for Ws in combinations(list(G.V - {Y})))


def MUCT(G: CD, Y: str) -> VSet:
    """ Minimal Unobserved Confounder's Territory """
    H = G[G.An(Y)]

    Qs = {Y}
    Ts = frozenset({Y})
    while Qs:
        Q1 = pop(Qs)
        Ws = CC(H, Q1)
        Ts |= Ws
        Qs = (Qs | H.de(Ws)) - Ts

    return Ts


def IB(G: CD, Y: str) -> VSet:
    # interventional_border
    Zs = MUCT(G, Y)
    return G.pa(Zs) - Zs


def MUCT_IB(G: CD, Y: str) -> Tuple[VSet, VSet]:
    return (Zs := MUCT(G, Y)), G.pa(Zs) - Zs


def POMISs(G: CD, Y: str, Ns: AbstractSet[str] = None) -> FrozenSet[VSet]:
    """ all POMISs for G with respect to Y

    Ns is a set of non-manipulable variables
    """
    if Ns is not None:
        assert Y not in Ns
        G = G.latent_projection(G.V - Ns)

    G = G[G.An(Y)]

    Ts, Xs = MUCT_IB(G, Y)
    H = G.do(Xs)[Ts | Xs]

    # note that the backward causal order is not necessary but any order is acceptable
    return subPOMISs(H, Y, only(H.causal_order(backward=True), Ts - {Y})) | {frozenset(Xs)}


def subPOMISs(G: CD, Y: str, Ws: List, obs=None) -> FrozenSet[VSet]:
    if obs is None:
        obs = set()

    out = []
    for i, W_i in enumerate(Ws):
        Ts, Xs = MUCT_IB(G.do({W_i}), Y)
        new_obs = obs | set(Ws[:i])
        if not (Xs & new_obs):
            out.append(Xs)

            if new_Ws := only(Ws[i + 1:], Ts):
                out.extend(subPOMISs(G.do(Xs)[Ts | Xs], Y, new_Ws, new_obs))

    return frozenset(as_fzsets(out))


def minimal_do(G: CD, Y: str, Xs: AbstractSet[str]) -> VSet:
    """ Non-redundant subset of Xs that entail the same E[Y|do(Xs)] """
    return frozenset(Xs & G.do(Xs).An(Y))
