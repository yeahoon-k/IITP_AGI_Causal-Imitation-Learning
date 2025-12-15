from typing import Sequence, FrozenSet, AbstractSet

from npsem.idty.zid import is_ID
from npsem.model import CausalDiagram

VSet = FrozenSet[str]
ASet = AbstractSet[str]


def CC(G: CausalDiagram):
    return G.c_components


def is_mzTR(G: CausalDiagram, Ys: ASet, Xs: ASet, Sss: Sequence[ASet], Zss: Sequence[ASet]) -> bool:
    """ mz-Transportability """
    assert Ys
    Vs = G.V  # alias
    assert all(Ss <= Vs for Ss in Sss)
    assert all(Zs <= Vs for Zs in Zss)

    if not Xs:
        return True

    if Vs - G.An(Ys):
        return is_mzTR(G[G.An(Ys)], Ys, Xs & G.An(Ys), [Ss & G.An(Ys) for Ss in Sss], [Zs & G.An(Ys) for Zs in Zss])

    if Ws := (Vs - Xs) - G.do(Xs).An(Ys):
        return is_mzTR(G, Ys, Xs | Ws, Sss, Zss)

    if len(CCs := CC(G - Xs)) > 1:
        return all(is_mzTR(G, S_i, Vs - S_i, Sss, Zss) for S_i in CCs)
    else:
        S, = CCs
        if CC(G) == {Vs}:
            for i in range(len(Sss)):
                if Sss[i].isdisjoint((G - Xs).V) and Zss[i] & Xs:
                    if is_ID(G - (Zss[i] & Xs), Ys, Xs - (Zss[i] & Xs)):
                        return True

            return False
        else:
            if S in CC(G):
                return True
            for S__ in CC(G):
                if S < S__:
                    return is_mzTR(G[S__], Ys, Xs & S__, [Ss & S__ for Ss in Sss], [Zs & S__ for Zs in Zss])

    raise AssertionError('impossible / coding error')


def is_TR(G: CausalDiagram, Ys: ASet, Xs: ASet, Ss: ASet) -> bool:
    """ Transportability (all possible experiments in the source, observation in the target) """
    return is_mzTR(G, Ys, Xs, [Ss], [G.V])


def is_mTR(G: CausalDiagram, Ys: ASet, Xs: ASet, Sss: Sequence[ASet]) -> bool:
    """ m-Transportability (all experiments in the sources, observation in the target) """
    return is_mzTR(G, Ys, Xs, Sss, [G.V] * len(Sss))


def is_zTR(G: CausalDiagram, Ys: ASet, Xs: ASet, Ss: ASet, Zs: ASet) -> bool:
    """ z-Transportability (experiments in the source, observation in the target) """
    return is_mzTR(G, Ys, Xs, [Ss], [Zs])
