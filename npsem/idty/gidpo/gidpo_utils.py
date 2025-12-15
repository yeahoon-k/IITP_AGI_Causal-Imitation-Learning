from typing import Tuple, Optional, FrozenSet, Union, AbstractSet

from scipy.special import binom

from npsem.model import CD

VSet = FrozenSet[str]
ASet = AbstractSet[str]

try:
    from rpy2 import robjects
    from rpy2.robjects.packages import importr

    dosearch_rpkg = importr('dosearch')
except ImportError as e:
    dosearch_rpkg = None
    robjects = None
    importr = None
    pass


def flipping_prob(p: float, n: int) -> float:
    s = 0.
    for i in range(n + 1):
        if i % 2 == 1:
            s += binom(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return s


def LP(G: CD, to_keep: VSet) -> CD:
    """ Latent Projection """
    assert to_keep <= G.V
    if to_keep == G.V:
        return G
    return G.marginalize_out(G.V - to_keep)


def MO(G: CD, to_out: ASet) -> CD:
    """ Marginalize-out """
    assert to_out <= G.V
    if to_out:
        return G.marginalize_out(frozenset(to_out))
    else:
        return G


def __query_string_for_dosearch(Ys: VSet, Xs: VSet = frozenset(), separator=','):
    """ String representation of a probability term (variable-based) """
    if Xs:
        qstr = f'P({separator.join(sorted(Ys))} | do({separator.join(sorted(Xs))}))'
    else:
        qstr = f'P({separator.join(sorted(Ys))})'
    return qstr


def dosearch_prepare(G, Ys, Xs, Zss, Vss):
    graph = '\n'.join([f' {x} -> {y} ' for x, y in G.edges] +
                      [f' {x} -- {y} ' for x, y, _ in G.biedges3])

    def do_term(Zs_):
        return f'|do({",".join(Zs_)})' if Zs_ else ''

    data = '\n'.join([f'p({",".join(Vs)}{do_term(Zs)})' for Zs, Vs in zip(Zss, Vss)])

    query = __query_string_for_dosearch(Ys, Xs)

    return graph, data, query


def dosearch_gidpo(G: CD, Ys: VSet, Xs: VSet, Zss: Tuple[VSet], Vss: Tuple[VSet],
                   formula=False) -> Union[bool, Tuple[bool, Optional[str]]]:
    """ Calling Tikka et al. """
    return dosearch_gidpo2(*dosearch_prepare(G, Ys, Xs, Zss, Vss), formula)


def dosearch_gidpo2(graph, data, query, formula=False) -> Union[bool, Tuple[bool, Optional[str]]]:
    """ Calling Tikka et al. """
    if dosearch_rpkg is None:
        raise ImportError('requires rpy2 (python package) and dosearch (R package).')
    if formula:
        out = dosearch_rpkg.dosearch(data, query, graph, control=robjects.r('list(formula=TRUE)'))
        if out[0][0]:
            return out[0][0], out[1][0]
        else:
            return out[0][0], None
    else:
        out = dosearch_rpkg.dosearch(data, query, graph)
        return out[0][0]
