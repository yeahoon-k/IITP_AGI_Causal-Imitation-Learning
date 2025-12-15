from typing import Collection, Sequence, List, AbstractSet, Set, Tuple

from npsem.idty.exp_rewrite import rewriter
from npsem.idty.gid import is_subID, subID, essential_conditions
from npsem.idty.idty_utils import NotIdentifiableError
from npsem.idty.prob_eq import EqNode, assign, Pr, ProbDistr, Π, Σ, frac_node
from npsem.model import CD
from npsem.utils import fzset_union

"""
References
----------
Sanghack Lee, Juan D. Correa, and Elias Bareinboim, 
Generalized Transportability: Synthesis of Experiments from Heterogeneous Domains 
In Proceedings of Thirty-fourth Conference on Artificial Intelligence (AAAI 2020)
"""

ASet = AbstractSet[str]


def CC(G: CD):
    return G.c_components


# noinspection NonAsciiCharacters
def selection_diagram(G: CD, Δ: Sequence[ASet]) -> Tuple[CD, List[Set[str]]]:
    """ Selection diagram and selection variables per domain """
    all_delta = sorted(fzset_union(Δ) & G.V)

    selection_variables = [f'S_{V}' for V in all_delta if V in G.V]

    H = CD(set(all_delta) | set(selection_variables), set(zip(selection_variables, all_delta)))
    SD = G + H
    SSs = [{f'S_{V}' for V in Delta_i if V in G.V} for Delta_i in Δ]

    return SD, SSs


# noinspection NonAsciiCharacters
def is_gTRC(G: CD, Ys: ASet, Xs: ASet, Ws: ASet, Zss: Sequence[Collection[ASet]], Δ: Sequence[ASet]) -> bool:
    """ see `gTRC` """
    assert len(Zss) == len(Δ)
    Ws, Xs = essential_conditions(G, Ys, Xs, Ws)
    return is_gTR(G, Ys | Ws, Xs, Zss, Δ)


# noinspection NonAsciiCharacters
def gTRC(G: CD, Ys: ASet, Xs: ASet, Ws: ASet, Zss: Sequence[Collection[ASet]], Δ: Sequence[ASet]) -> EqNode:
    """
    Sanghack Lee, Juan D. Correa, and Elias Bareinboim
    Generalized Transportability: Synthesis of Experiments from Heterogeneous Domains
    In Proceedings of Thirty-fourth Conference on Artificial Intelligence (AAAI 2020)
    """
    assert len(Zss) == len(Δ)
    Ws, Xs = essential_conditions(G, Ys, Xs, Ws)
    if Ws:
        return frac_node(f := rewriter(gTR(G, Ys | Ws, Xs, Zss, Δ), G),
                         Σ(Ys, f))
    else:
        return gTR(G, Ys, Xs, Zss, Δ)


# noinspection NonAsciiCharacters
def is_gTR(G: CD, Ys: ASet, Xs: ASet, Zss: Sequence[Collection[ASet]], Δ: Sequence[ASet]) -> bool:
    assert not Δ[0], 'The first element is the target domain, which cannot be different from itself.'
    assert len(Zss) == len(Δ)
    SD, SSs = selection_diagram(G, Δ)
    Vs = G.V
    SDXs = (SD - Xs)

    for i, Zss_i in enumerate(Zss):
        if Δ[i].isdisjoint(Ys) and SDXs.independent(Ys, SSs[i]):  # short circuiting
            for Zs in Zss_i:
                if Xs == Zs & Vs:
                    return True

    if Vs != G.An(Ys):
        return is_gTR(G[G.An(Ys)], Ys, Xs & G.An(Ys), Zss, Δ)

    if Ws := (Vs - Xs) - G.do(Xs).An(Ys):
        return is_gTR(G, Ys, Xs | Ws, Zss, Δ)

    if len(CCs := CC(G - Xs)) > 1:
        return all(is_gTR(G, S_i, Vs - S_i, Zss, Δ) for S_i in sorted(CCs))

    for i, Zss_i in enumerate(Zss):
        if not Δ[i].isdisjoint(Ys) or SDXs.dependent(Ys, SSs[i]):
            continue
        for Zs in Zss_i:
            if Zs & Vs <= Xs:
                if is_subID(G - (Zs & Vs), Ys, Xs - Zs):
                    return True
    return False


# noinspection NonAsciiCharacters
def gTR(G: CD, Ys: ASet, Xs: ASet, Zss: Sequence[Collection[ASet]], Δ: Sequence[ASet]) -> EqNode:
    Vs = G.V
    SD, SSs = selection_diagram(G, Δ)

    for i, Zss_i in enumerate(Zss):
        if (SD - Xs).independent(Ys, SSs[i]):
            for Zs in Zss_i:
                if Xs == Zs & Vs:
                    return assign(Pr(ProbDistr(Zs, G.V, domain_id=i), Ys), Zs - Vs)

    if Vs != G.An(Ys):
        return gTR(G[G.An(Ys)], Ys, Xs & G.An(Ys), Zss, Δ)

    if Ws := (Vs - Xs) - G.do(Xs).An(Ys):
        return assign(gTR(G, Ys, Xs | Ws, Zss, Δ), Ws)

    if len(CCs := CC(G - Xs)) > 1:
        return Σ(Vs - (Xs | Ys), Π([gTR(G, S_i, Vs - S_i, Zss, Δ) for S_i in sorted(CCs)]))

    for i, Zss_i in enumerate(Zss):
        if (SD - Xs).dependent(Ys, SSs[i]):
            continue
        for Zs in Zss_i:
            if Zs & Vs <= Xs:
                if (formula := assign(subID(G - (Zs & Vs), Ys, Xs - Zs, ProbDistr(Zs, Vs, domain_id=i)), Zs - Vs)) is not None:
                    return formula

    raise NotIdentifiableError('not identifiable', None)  # TODO sthicket
