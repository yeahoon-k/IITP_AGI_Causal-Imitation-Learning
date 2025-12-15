from typing import Set, AbstractSet, Dict, Sequence

from npsem.model import CausalDiagram as CD
from npsem.utils import sort_with

ASet = AbstractSet[str]

def before(X, order: Sequence[str]):
    return set(order[:order.index(X)])

def hasValidAdjustment(G: CD, Ox: ASet, Oi: str, Xi: str, Os: ASet = None, Y='Y', order=None):
    """ Sequential Imitation Causal Learning 2021 """
    if Os is None:
        Os = G.V
    if order is None:
        order = G.causal_order()
    GY = G[G.An(Y)]
    H = GY.latent_projection(Os & GY.V)
    Cs = H.c_component(Oi)
    GC = GY[GY.do(H.pa(Cs)).An(Cs)]
    OC = Cs - (Ox | {Oi})

    return GC.independent(Oi, OC, OC & before(Xi, order) & GC.V)


def findOx(G: CD, Os: Set[str], Xs: set, Y: str, order=None) -> ASet:
    """ Sequential Imitation Causal Learning 2021 """
    if order is None:
        order = G.causal_order()

    GY = G[G.An(Y)]  # ancestral to Y
    H = GY.latent_projection(Os & GY.V)  # retain only observables

    Ox = dict()
    while True:
        pastOx = dict(Ox)
        # reverse temporal order, observables
        for Oi in H.causal_order(backward=True):
            # ch+ = H.ch
            if H.ch(Oi) and H.ch(Oi) <= Ox.keys():
                # in temporal order,
                Xi = sort_with({Ox[_] for _ in H.ch(Oi)}, H.causal_order())[0]
                if hasValidAdjustment(G, set(Ox.keys()), Oi, Xi, Os, order=order):
                    Ox[Oi] = Xi
            elif Oi in Xs and hasValidAdjustment(GY, set(Ox.keys()), Oi, Oi, Os, order=order):
                Ox[Oi] = Oi
        if pastOx == Ox:
            break

    return set(Ox.keys())


def markov_boundary(G: CD, Ox: ASet, Os: ASet) -> ASet:
    H = G.latent_projection(Os & G.V)
    return H.Pa(H.c_component(H.Ch(Ox))) - Ox


def boundary_actions(G: CD, Ox: ASet, Os: ASet, Xs: ASet) -> ASet:
    H = G.latent_projection(Os & G.V)
    return {Xi for Xi in Xs & Os if not (H.ch(Xi) <= Ox)}


def seq_imi_graphs():
    return {'fig2a': (CD('X->Z->Y, X->Y, X<->Z'), {'X'}, {'Y'}, None),
            'fig2b': (CD('X1->X2->Y<-X1<->X2'), {'X1', 'X2'}, {'Y'}, None),
            'fig2c': (CD('Z<->X1->W->X2->Y<-Z'), {'X1', 'X2'}, {'Y'}, ['X1', 'Z', 'W', 'X2', 'Y']),
            'fig2d': (CD('X1<->Z->W->X2->Y<-X1, W<->Y'), {'X1', 'X2'}, {'Y'}, None),

            'tab1-1': (CD('X1->X2->Y<->Z, X1<->Z<->X2'), {'X1', 'X2'}, {'Y'}, ['Z1', 'X1', 'X2', 'Y']),
            'tab1-2': (CD('X1->X2->Y<-Z<->X1'), {'X1', 'X2'}, {'Y'}, ['Z', 'X1', 'X2', 'Y']),
            'tab1-3': (CD('X1->X2->Y<-Z<->X1'), {'X1', 'X2'}, {'Y'}, ['X1', 'Z', 'X2', 'Y']),
            'tab1-4': (CD('X1<->Z->X2->Y<-X1, Z<->Y'), {'X1', 'X2'}, {'Y'}, ['X1', 'Z', 'X2', 'Y']),
            }  # type: Dict[name, Tuple[CD, ASet, ASet, Optional[List[str]]]]


class Policy(Dict[str, ASet]):
    pass


def seq_pi_bd(G: CD, Xs: ASet, Y: str, order=None) -> Policy:
    if order is None:
        order = G.causal_order()
    Ox = findOx(G, G.V, Xs, Y, order)
    Xs_prime = Ox & Xs
    Zs = markov_boundary(G[G.An(Y)].underbar(Xs_prime), Ox, G.V)
    XBs = boundary_actions(G, Ox, G.V, Xs)
    policy = {Xi: (Zs | XBs) & before(Xi, order) for Xi in Xs_prime}
    return policy


