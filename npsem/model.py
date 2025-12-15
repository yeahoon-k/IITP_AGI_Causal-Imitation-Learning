import functools
import itertools
from collections import defaultdict, deque
from itertools import product, chain
from typing import Dict, Iterable, Optional, Sequence, AbstractSet, Union, Collection, List, Callable, Any, TypeVar, KeysView
from typing import FrozenSet, Tuple

import networkx as nx
import numpy as np
from matplotlib.patches import ConnectionStyle, FancyArrowPatch
from scipy.optimize import minimize

from npsem.parsers import parse_scm_functions, parse_graph
from npsem.utils import fzset_union, set_union, sortup, sortup2, with_default, shuffled, dict_except, dict_only, first_diff_index, sort_with, as_sortups, sl_group_by, pairs, visited_queue

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')
VSet = FrozenSet[str]
ASet = AbstractSet[str]
ASet2 = Union[str, AbstractSet[str]]
VVs = Union[str, Iterable[str]]

# ASCII
RIGHT_ARROW = "->"
LR_ARROW = "<->"


def exo_independent(G: 'CausalDiagram', vxs: ASet2, uxs: ASet2,
                    vys: ASet2, uys: ASet2,
                    zs: ASet2, uws: ASet2 = frozenset(), uzs: ASet2 = frozenset()):
    """ Conditional independence test with exogenous variables """
    # "Conditional Independence with Exogenous Variables" in Sanghack's overleaf
    # values are all endogenous variables
    # e.g., if uxs = {A,B,C}, then {\UU_A, \UU_B, \UU_C}
    vxs, uxs, vys, uys, zs, uws, uzs = [_wrap(_) for _ in [vxs, uxs, vys, uys, zs, uws, uzs]]

    uzs -= uws
    uxs -= uws
    uys -= uws

    # given
    vxs -= zs
    vys -= zs
    # deterministic
    vxs -= {x for x in vxs if G.pa(x) <= zs and x in uws}
    vys -= {y for y in vys if G.pa(y) <= zs and y in uws}

    H = G + {(_1, _2, f'U_{_1}_{_2}') for _1, _2 in pairs(G.confounded_withs(uws | uzs)) if not G.is_confounded(_1, _2)}
    H -= H.UCs(uws)  # more like labels for bidirected edges not necessarily an unobserved confoudner
    H -= {w for w in uws if G.pa(w) <= zs}

    return H.independent(vxs, vys, zs) and \
           _exo_independent_uv(H, uys, vxs, zs) and \
           _exo_independent_uv(H, uxs, vys, zs) and \
           _exo_independent_uu(H, uxs, uys, zs)


def _exo_independent_uv(G: 'CausalDiagram', uxs: ASet, ys: ASet, zs: ASet):
    G = G.underbar(uxs & zs)
    G = G.directed_overbar(uxs - G.An(zs))
    return G.independent(uxs, ys, zs - uxs)


def _exo_independent_uu(G: 'CausalDiagram', uxs: ASet, uys: ASet, zs: ASet):
    G = G.underbar((uxs | uys) & zs)
    G = G.directed_overbar((uxs | uys) - G.An(zs))
    return G.independent(uxs, uys, zs - (uxs | uys))


def _draw_helper(pos, edges, biedges):
    if not biedges:
        return []

    points = [(x, y) for x, y in pos.values()] + [(0.5 * pos[a][0] + 0.5 * pos[b][0], 0.5 * pos[a][1] + 0.5 * pos[b][1]) for a, b in edges]
    biedges2points = [((pos[va][0], pos[va][1]), (pos[vb][0], pos[vb][1])) for va, vb in biedges]

    # negative distance
    cur_dist = 0
    cur_rads = None
    for _ in range(3):
        neg_dist, rads = _draw_helper_inner(points, biedges2points)
        if cur_dist < -neg_dist:
            cur_dist = -neg_dist
            cur_rads = list(rads)

    return cur_rads


def _draw_helper_inner(points, biedges2points):
    def total_distance(rads):
        # midpoint = 0.5 * (xa + xb), 0.5 * (ya + yb)
        # slope = (xb - xa), (yb - ya)  # a to b
        # rad_direction = -(yb-ya), (xb-xa)  # (x,y) -> (y,-x)
        control_points = [(0.5 * (xa + xb) + rad_ab * (yb - ya), 0.5 * (ya + yb) - rad_ab * (xb - xa))
                          for rad_ab, ((xa, ya), (xb, yb)) in zip(rads, biedges2points)]

        dist = 0.
        for xb, yb in control_points:
            # TODO exclude the starting and end points ...
            dist += sum([np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2) for xa, ya in points])
        if len(control_points) > 1:
            for i, (xa, ya) in enumerate(control_points):
                dist += 2 * sum([np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2) for xb, yb in control_points[:i] + control_points[i + 1:]])

        return -dist

    res = minimize(total_distance,
                   np.random.rand(len(biedges2points)) * 0.8 - 0.4,
                   method='SLSQP',
                   bounds=((-0.4, .4),) * len(biedges2points))

    assert res.success
    return res.fun, list(res.x)


def _paths_and_bipaths(G: 'CausalDiagram') -> Tuple[List[List[str]], List[List[str]]]:
    """ Multiple simple paths representing directed edges and bidirected edges """
    # find a longest directed path and remove edges contained the path. repeat.
    nxG = nx.DiGraph(sortup(G.edges))
    paths = []
    while nxG.edges:
        # noinspection PyTypeChecker
        path = nx.dag_longest_path(nxG)  # type: List[str]
        paths.append(path)
        for x, y in zip(path, path[1:]):
            nxG.remove_edge(x, y)

    # find a longest undirected path and remove corresponding bi-edges. repeat
    nxG = nx.Graph([(x, y) for x, y in G.u2vv.values()])
    bipaths = []
    while nxG.edges:
        temppaths = []
        for x, y in itertools.combinations(sortup(nxG.nodes), 2):
            for spath in nx.all_simple_paths(nxG, x, y):
                temppaths.append(spath)
        selected = sorted(temppaths, key=lambda _spath: len(_spath), reverse=True)[0]
        bipaths.append(selected)
        for x, y in zip(selected, selected[1:]):
            nxG.remove_edge(x, y)

    return paths, bipaths


def _pairs2dict(xys, backward=False):
    dd = defaultdict(set)
    if backward:
        for x, y in xys:
            dd[y].add(x)
    else:
        for x, y in xys:
            dd[x].add(y)

    return defaultdict(frozenset, {key: frozenset(vals) for key, vals in dd.items()})


def _wrap(v_or_vs: Union[str, Iterable[str]]) -> Optional[VSet]:
    if v_or_vs is None:
        return None
    if isinstance(v_or_vs, str):
        return frozenset({v_or_vs})
    else:
        return frozenset(v_or_vs)


class CausalDiagram:
    def __init__(self,
                 vs: Optional[Union[Iterable[str], str]] = None,
                 directed_edges: Optional[Iterable[Tuple[str, str]]] = frozenset(),
                 bidirected_edges: Optional[Iterable[Tuple[str, str, str]]] = frozenset(),
                 *,
                 copy_from: 'CausalDiagram' = None,
                 with_do: Optional[ASet] = None,
                 with_induced: Optional[ASet] = None,
                 parse_kwargs=None):
        """
        Causal diagram represented as directed and bidirected edges.

        Parameters
        ----------
        vs:
            A (sub)set of endogenous variables.
            Since other variables appearing in the edges will be added, this parameter is only necessary when there exists a varaible without any connection (directed or bidirected)
        directed_edges:
            Directed edges represented as a pair of two variables.
        bidirected_edges:
            Bidirected edges represented as a tuple of variable 1, variable 2, and label. (variable 1 must be smaller than variable 2)
        copy_from:
            a causal diagram from which new causal diagram is induced
        with_do:
            variables applying do-operation over `copy_from`
        with_induced:
            variables applying subgraph over `copy_from`

        """
        vs = with_default(vs, set())
        if isinstance(vs, str):
            vs, directed_edges, bidirected_edges = parse_graph(vs, **with_default(parse_kwargs, {}))

        with_do = _wrap(with_do)
        with_induced = _wrap(with_induced)
        if with_do and with_induced:
            raise ValueError(f'{with_do=} and {with_induced=} cannot be simultaneously used at this point')

        # V, U, u2vv, _pa, _ch, _an, _de
        if copy_from is not None:
            if with_do is not None:
                self.__do_from(copy_from, with_do)
            elif with_induced is not None:
                self.__induced_from(copy_from, with_induced)
            else:
                self.__copy_from(copy_from)
        else:
            directed_edges = list(directed_edges)
            bidirected_edges = list(bidirected_edges)
            self.V = frozenset(vs) | fzset_union(directed_edges) | fzset_union((x, y) for x, y, _ in bidirected_edges)  # type: VSet
            if not all(v for v in self.V):
                raise ValueError('An empty name exists in observed variables.')

            self.U = frozenset(u for _, _, u in bidirected_edges)
            if not all(u for u in self.U):
                raise ValueError('An empty name exists in unobserved variables.')

            # dictionary for unobserved variables, stores two endogenous children as frozenset for each unobserved variable.
            self.u2vv = {u: frozenset({x, y}) for x, y, u in bidirected_edges}

            self._ch = _pairs2dict(directed_edges)
            self._pa = _pairs2dict(directed_edges, backward=True)
            self._an = dict()  # cache
            self._de = dict()  # cache
            assert self._ch.keys() <= self.V and self._pa.keys() <= self.V

        # rest of things
        self.edges = tuple((x, y) for x, ys in self._ch.items() for y in ys)  # type: Tuple[Tuple[str, str], ...]
        # TODO reuse the causal order from `copy_from`: e.g., a subgraph operation, or only bidirected edges are changed, or projection?!
        self.causal_order = functools.lru_cache(maxsize=2)(self.causal_order)
        self._do_ = functools.lru_cache()(self._do_)
        self.__marginalize_out = functools.lru_cache()(self.__marginalize_out)
        self.marginalize_out = functools.lru_cache()(self.marginalize_out)
        self.__underbar = functools.lru_cache()(self.__underbar)
        # TODO faster
        self.__cc = None
        self.__cc_dict = None
        self.__h = None
        self.__characteristic = None
        self.__confoundeds = None
        _u_pas = defaultdict(set)
        for u, xy in self.u2vv.items():
            for v in xy:
                _u_pas[v].add(u)
        self.u_pas = defaultdict(frozenset, {v: frozenset(us) for v, us in _u_pas.items()})
        self.__biedges3 = None  # cached result for biedges3

    def __copy_from(self, copy_from: 'CausalDiagram'):
        self.V = copy_from.V  # type: VSet
        self.U = copy_from.U  # type: VSet
        self.u2vv = copy_from.u2vv
        self._ch = copy_from._ch
        self._pa = copy_from._pa
        self._an = copy_from._an
        self._de = copy_from._de

    def __induced_from(self, copy_from: 'CausalDiagram', with_induced: VSet):
        if not with_induced <= copy_from.V:
            raise ValueError(f'{with_induced} not a subset of {copy_from.V}')
        # assert with_induced <= copy_from.V
        removed = copy_from.V - with_induced
        self.V = with_induced
        self.u2vv = {u: val for u, val in copy_from.u2vv.items() if val <= self.V}
        self.U = _wrap(self.u2vv)
        children_are_removed = copy_from.pa(removed) & self.V
        parents_are_removed = copy_from.ch(removed) & self.V
        ancestors_are_removed = copy_from.de(removed) & self.V
        descendants_are_removed = copy_from.an(removed) & self.V
        self._pa = defaultdict(frozenset, {x: (copy_from._pa[x] - removed) if x in parents_are_removed else copy_from._pa[x] for x in self.V})
        self._ch = defaultdict(frozenset, {x: (copy_from._ch[x] - removed) if x in children_are_removed else copy_from._ch[x] for x in self.V})
        self._an = dict_only(copy_from._an, self.V - ancestors_are_removed)
        self._de = dict_only(copy_from._de, self.V - descendants_are_removed)

    def __do_from(self, copy_from: 'CausalDiagram', with_do: ASet):
        self.V = copy_from.V  # type: VSet
        self.U = _wrap(u for u in copy_from.U if with_do.isdisjoint(copy_from.u2vv[u]))  # type: VSet
        self.u2vv = {u: val for u, val in copy_from.u2vv.items() if u in self.U}
        # copy cautiously
        dopa = copy_from.pa(with_do)
        doAn = copy_from.An(with_do)
        doDe = copy_from.De(with_do)
        self._pa = defaultdict(frozenset, {k: frozenset() if k in with_do else v for k, v in copy_from._pa.items()})
        self._ch = defaultdict(frozenset, {k: (v - with_do) if k in dopa else v for k, v in copy_from._ch.items()})
        self._an = dict_except(copy_from._an, doDe)
        self._de = dict_except(copy_from._de, doAn)

    def UCs(self, v_or_vs: Union[str, Iterable[str]]) -> VSet:
        """ unobserved confounders connecting the given variables """
        if isinstance(v_or_vs, str):
            return self.u_pas[v_or_vs]
        else:
            return fzset_union(self.u_pas[v] for v in v_or_vs)

    def sinks(self) -> VSet:
        """ Variables that do not have children. """
        return frozenset(V for V in self.V if not self.ch(V))

    def sources(self) -> VSet:
        """ Variables that do not have parents. """
        return frozenset(V for V in self.V if not self.pa(V))

    def __contains__(self, item):
        """ Checks the membership of the given parameter

        If it is a string, check observed variables and unobserved variables.
        If it is a set of two items, check whether two items are confounded.
        If it is two items in other collection type, checks whether an edge from the first item to the second item exists
        If it is three items, say x, y, u, then check whether x and y are confounded with unobserved variable u.

        Raises an exception if the given item does not satisfy the above criteria.
        """
        if isinstance(item, str):
            return item in self.V or item in self.U
        if len(item) == 2:
            if isinstance(item, AbstractSet):
                x, y = item
                return self.is_confounded(x, y)
            else:
                return tuple(item) in self.edges
        if len(item) == 3:
            x, y, u = item
            return self.is_confounded(x, y) and u in self.u2vv and self.u2vv[u] == frozenset({x, y})
        raise ValueError(f'an unknown use case: {item}')

    def __lt__(self, other):
        """ If this graph is a proper subgraph of `other`

        Note: the names of unobserved confounders are ignored.
        """
        if not isinstance(other, CausalDiagram):
            return False
        return self <= other and self != other

    def __le__(self, other):
        """ If this graph is a subgraph of `other`

        Note: the names of unobserved confounders are ignored.
        """
        if not isinstance(other, CausalDiagram):
            return False
        return self.V <= other.V and set(self.edges) <= set(other.edges) and set(self.u2vv.values()) <= set(other.u2vv.values())

    def __ge__(self, other):
        """ If `other` is a subgraph of this graph

        Note: the names of unobserved confounders are ignored.
        """
        if not isinstance(other, CausalDiagram):
            return False
        return self.V >= other.V and set(self.edges) >= set(other.edges) and set(self.u2vv.values()) >= set(other.u2vv.values())

    def __gt__(self, other):
        """ If `other` is a proper subgraph of this graph

        Note: the names of unobserved confounders are ignored.
        """
        if not isinstance(other, CausalDiagram):
            return False
        return self >= other and self != other

    def Pa(self, v_or_vs: VVs) -> VSet:
        """ Parents of the argument (inclusive) """
        return self.pa(v_or_vs) | _wrap(v_or_vs)

    def pa(self, v_or_vs: VVs) -> VSet:
        """ Parents of the argument """
        if isinstance(v_or_vs, str):
            return self._pa[v_or_vs]
        else:
            return fzset_union(self._pa[v] for v in v_or_vs)

    def ch(self, v_or_vs: VVs) -> VSet:
        """ Children of the argument """
        if isinstance(v_or_vs, str):
            return self._ch[v_or_vs]
        else:
            return fzset_union(self._ch[v] for v in v_or_vs)

    def Ch(self, v_or_vs: VVs) -> VSet:
        """ Children of the argument (inclusive) """
        return self.ch(v_or_vs) | _wrap(v_or_vs)

    def An(self, v_or_vs: VVs) -> VSet:
        """ Ancestors of the argument (inclusive) """
        if isinstance(v_or_vs, str):
            return self.__an(v_or_vs) | {v_or_vs}
        return self.an(v_or_vs) | _wrap(v_or_vs)

    def an(self, v_or_vs: VVs) -> VSet:
        """ Ancestors of the argument """
        if isinstance(v_or_vs, str):
            return self.__an(v_or_vs)
        return fzset_union(self.__an(v) for v in _wrap(v_or_vs))

    def De(self, v_or_vs: VVs) -> VSet:
        """ Descendants of the argument (inclusive) """
        if isinstance(v_or_vs, str):
            return self.__de(v_or_vs) | {v_or_vs}
        return self.de(v_or_vs) | _wrap(v_or_vs)

    def de(self, v_or_vs: VVs) -> VSet:
        """ Descendants of the argument """
        if isinstance(v_or_vs, str):
            return self.__de(v_or_vs)
        return fzset_union(self.__de(v) for v in _wrap(v_or_vs))

    def __an(self, v: str) -> VSet:
        if v in self._an:
            return self._an[v]
        self._an[v] = fzset_union(self.__an(parent) for parent in self._pa[v]) | self._pa[v]
        return self._an[v]

    def __de(self, v: str) -> VSet:
        if v in self._de:
            return self._de[v]
        self._de[v] = fzset_union(self.__de(child) for child in self._ch[v]) | self._ch[v]
        return self._de[v]

    def NonDe(self, v_or_vs: VVs) -> VSet:
        return self.V - self.De(v_or_vs)

    def do(self, v_or_vs: VVs) -> 'CausalDiagram':
        """ Causal diagram with edges incoming to the argument removed """
        return self._do_(_wrap(v_or_vs))

    def _do_(self, v_or_vs) -> 'CausalDiagram':
        return CausalDiagram(None, None, None, copy_from=self, with_do=_wrap(v_or_vs))

    def has_edge(self, x: str, y: str) -> bool:
        """ Whether a directed edge `x` -> `y` exists """
        return y in self._ch[x]

    def is_confounded(self, x: str, y: str) -> bool:
        """ Whether a bidirected edge `x` <-> `y` exists """
        self.__ensure_confoundeds_cached()
        if x not in self.__confoundeds:
            return False
        return y in self.__confoundeds[x]

    def u_of(self, x: str, y: str) -> Optional[str]:
        """ an unobserved confounder connecting `x` and `y` if exists, else `None` """
        key = {x, y}
        for u, ab in self.u2vv.items():
            if ab == key:
                return u
        return None

    def confounds(self, u: str) -> VSet:
        """ A set of two variables connected by the given UC. """
        if not (isinstance(u, str) and u in self.U):
            raise ValueError(f"{u} does not exist")
        return self.u2vv[u]

    def confounded_withs(self, v_or_vs: VVs) -> VSet:
        """ Variables confounded with the given argument """
        self.__ensure_confoundeds_cached()
        if isinstance(v_or_vs, str):
            return self.__confoundeds[v_or_vs]
        else:
            return fzset_union(self.__confoundeds[w] for w in v_or_vs)

    def __getitem__(self, item):
        """ Vertex-induced subgraph """
        return self.induced(item)

    def induced(self, v_or_vs: VVs) -> 'CausalDiagram':
        """ Vertex-induced subgraph """
        v_or_vs = _wrap(v_or_vs)
        if v_or_vs - self.V:
            raise ValueError(f'Variables out-of-scope: {v_or_vs - self.V}.')
        if v_or_vs == self.V:
            return self
        return CausalDiagram(None, None, None, copy_from=self, with_induced=v_or_vs)

    def UCs_explicitized(self):
        """ Causal diagram with UCs transformed to V (observables) """
        return CausalDiagram(self.V | self.U, self.edges + tuple([(u, c) for u in self.U for c in self.confounds(u)]))

    @property
    def characteristic(self) -> Tuple:
        """ An integer which characterizes the topology of the graph.

        Useful if two graphs are equivalent under permutation. """
        if self.__characteristic is None:
            self.__characteristic = (len(self.V),
                                     len(self.edges),
                                     len(self.u2vv),
                                     sortup([(len(self.ch(v)), len(self.pa(v)), len(self.confounded_withs(v))) for v in
                                             self.V]))
        return self.__characteristic

    def edges_removed(self, edges_to_remove: Iterable[Sequence[str]], strict=True) -> 'CausalDiagram':
        """ Edge-subgraph with the provided edges removed

        Every element is an edge (a pair) or bidirected edge (a triple)
        Note that the name of unobserved variable should match
        """
        edges_to_remove = list(edges_to_remove)
        if not edges_to_remove:
            return self

        edges_to_remove = [tuple(edge) for edge in edges_to_remove]

        dir_edges = {edge for edge in edges_to_remove if len(edge) == 2}
        bidir_edges = {edge for edge in edges_to_remove if len(edge) == 3}

        assert len(dir_edges) + len(bidir_edges) == len(edges_to_remove)

        bidir_edges = frozenset((*sorted([x, y]), u) for x, y, u in bidir_edges)

        if strict:
            if not dir_edges <= set(self.edges):
                raise ValueError(f'unknown edges: {dir_edges - set(self.edges)}')
            if not bidir_edges <= set(self.biedges3):
                # noinspection PyTypeChecker
                raise ValueError(f'unknown edges: {bidir_edges - set(self.biedges3)}')

        # noinspection PyTypeChecker
        return CausalDiagram(self.V, set(self.edges) - dir_edges, self.biedges3 - bidir_edges)

    def without_UCs(self) -> 'CausalDiagram':
        """ With all bidirected edges removed """
        # return self.edges_removed(self.biedges3)
        return CausalDiagram(self.V, self.edges)

    def without_directeds(self) -> 'CausalDiagram':
        """ With all directed edges removed """
        # return self.edges_removed(self.edges)
        return CausalDiagram(self.V, bidirected_edges=self.biedges3)

    def __copy__(self):
        return CausalDiagram(self.V, set(self.edges), self.biedges3)

    def __deepcopy__(self, memodict=None):
        return CausalDiagram(self.V, set(self.edges), self.biedges3)

    def __sub__(self, v_or_vs_or_edges: Union[str, Iterable[str], Iterable[Sequence[str]]]) -> 'CausalDiagram':
        """ Remove elements of the graph

        str or a collection of str will be treated as (un)observed variables.
        Any other types of data will be delegated to `edges_removed`
        """
        if not v_or_vs_or_edges:
            return self

        if isinstance(v_or_vs_or_edges, str):
            v_or_vs_or_edges = [v_or_vs_or_edges]

        vs = list(v_or_vs_or_edges)
        if isinstance(vs[0], str):
            vs = set(vs)
            assert all(isinstance(_, str) for _ in vs), f'cannot be mixed with string and other types, {v_or_vs_or_edges}'
            unknowns = vs - self.V - self.U
            assert not unknowns, f'unknown elements: {unknowns}'

            delVs = vs & self.V
            delUs = vs & self.U
            H = self.edges_removed([(*sorted(self.confounds(u)), u) for u in delUs])
            return H[H.V - delVs]

        # assert set(v_or_vs_or_edges) <= set(self.edges)
        return self.edges_removed(v_or_vs_or_edges)

    def causal_order(self, backward=False) -> Tuple[str, ...]:
        """ The ordered variables from top (sources) to bottom (sink)

        Note: generally not uniquely determined, but the result will be cached and the same results will be returned for the same graph object.
        """
        # TODO native method without networkx
        gg = nx.DiGraph(self.edges)
        gg.add_nodes_from(self.V)
        top_to_bottom = list(nx.topological_sort(gg))
        if backward:
            return tuple(reversed(top_to_bottom))
        else:
            return tuple(top_to_bottom)

    def __or__(self, other):
        """ see `__add__` """
        return self.__add__(other)

    def __add__(self, edges_or_v_or_vs) -> 'CausalDiagram':
        """ Causal diagram with new elements added

        If the argument is
            a causal diagram, then merge them. see `merge_two_cds`,
            a string or a collection of strings, then treat it as a variable.
        Otherwise,
            it is considered as a collection of edges (either directed (a pair) or bidirected (a triple))

        Note:
            Currently the existence of cycle is not checked. (calling `An` may throw a 'maximum recursion depth exceeded error'.)
        """
        # TODO check a cycle
        if isinstance(edges_or_v_or_vs, CausalDiagram):
            return merge_two_cds(self, edges_or_v_or_vs)

        vs, edges = None, None
        if isinstance(edges_or_v_or_vs, str):
            vs = {edges_or_v_or_vs}
        elif all(isinstance(elem, str) for elem in edges_or_v_or_vs):
            vs = set(edges_or_v_or_vs)
        else:
            edges = edges_or_v_or_vs

        if vs is not None:
            return CausalDiagram(self.V | vs, set(self.edges), self.biedges3)

        directed_edges = {edge for edge in edges if len(edge) == 2}
        bidirected_edges = {edge for edge in edges if len(edge) == 3}

        # check
        for x, y, u in bidirected_edges:
            # try to add x<-u->y
            assert u is not None
            # if u already exists, should match x<-u->y
            if u in self.U:
                if self.confounds(u) != {x, y}:
                    raise ValueError(f'{u} already confounds {self.confounds(u)} and cannot create {x}<-{u}->{y}')
            # if x and y are already confounded, it must be confounded with the same name, x<-u->y
            if self.is_confounded(x, y):
                if (org_u := self.u_of(x, y)) != u:
                    raise ValueError(f'{x} and {y} are already confounded with {org_u} (vs. {u})')

        return CausalDiagram(self.V, set(self.edges) | directed_edges, self.biedges3 | bidirected_edges)

    def __ensure_confoundeds_cached(self):
        if self.__confoundeds is None:
            self.__confoundeds = dict()
            for u, (x, y) in self.u2vv.items():
                if x not in self.__confoundeds:
                    self.__confoundeds[x] = set()
                if y not in self.__confoundeds:
                    self.__confoundeds[y] = set()
                self.__confoundeds[x].add(y)
                self.__confoundeds[y].add(x)
            self.__confoundeds = {x: frozenset(ys) for x, ys in self.__confoundeds.items()}
            for v in self.V:
                if v not in self.__confoundeds:
                    self.__confoundeds[v] = frozenset()

    def __ensure_cc_cached(self):
        # TODO faster
        if self.__cc is None:
            self.__ensure_confoundeds_cached()
            ccs = []
            remain = set(self.V)
            found = set()
            while remain:
                v = next(iter(remain))
                a_cc = set()
                to_expand = [v]
                while to_expand:
                    v = to_expand.pop()
                    a_cc.add(v)
                    to_expand += list(self.__confoundeds[v] - a_cc)
                ccs.append(a_cc)
                found |= a_cc
                remain -= found
            self.__cc2 = frozenset(frozenset(a_cc) for a_cc in ccs)
            self.__cc_dict2 = {v: a_cc for a_cc in self.__cc2 for v in a_cc}

            self.__cc = self.__cc2
            self.__cc_dict = self.__cc_dict2

    @property
    def c_components(self) -> FrozenSet[VSet]:
        """ C-component decomposition of V """
        self.__ensure_cc_cached()
        return self.__cc

    @property
    def is_one_cc(self):
        """ Whether the graph is a single c-component """
        return len(self.c_components) == 1

    def c_component(self, v_or_vs: VVs) -> VSet:
        """ Union of c-components containing the given argument """
        self.__ensure_cc_cached()
        return fzset_union(self.__cc_dict[v] for v in _wrap(v_or_vs))

    @property
    def biedges3(self) -> FrozenSet[Tuple[str, str, str]]:
        """ Bidirected edges as a set of triples

        Note: the first two elements (confounded observed variables) are sorted
        """
        # noinspection PyTypeChecker
        if self.__biedges3 is None:
            self.__biedges3 = frozenset((*sorted([x, y]), u) for u, (x, y) in self.u2vv.items())
        # noinspection PyTypeChecker
        return self.__biedges3

    def __eq__(self, other):
        """ Check whether two causal diagrams are equal without checking the names for their UCs """
        if not isinstance(other, CausalDiagram):
            return False
        if self.V != other.V:
            return False
        if set(self.edges) != set(other.edges):
            return False
        if set(self.u2vv.values()) != set(other.u2vv.values()):  # does not care about U's name
            return False
        return True

    def directed_overbar(self, v_or_vs: VVs):
        """ Remove directed edges onto the variables """
        return self - {(_, v) for v in _wrap(v_or_vs) for _ in self.pa(v)}

    def underbar(self, v_or_vs: VVs) -> 'CausalDiagram':
        """ With outgoing edges from the given arguments removed """
        return self.__underbar(_wrap(v_or_vs))

    def __underbar(self, vs: VSet) -> 'CausalDiagram':
        return self - {(v, chv) for v in vs for chv in self.ch(v)}

    under = underbar

    def __hash__(self):
        if self.__h is None:
            self.__h = hash(sortup(self.V)) ^ hash(sortup(self.edges)) ^ hash(sortup2(self.u2vv.values()))
        return self.__h

    def __independent(self, x: str, y: str, zs: VSet = frozenset()) -> bool:
        # TODO efficient
        assert x not in zs
        assert y not in zs
        assert x != y

        colliderables = set_union(self.An(z) for z in zs)
        chs = {('>', ch) for ch in self.ch(x)}
        pas = {('<', pa) for pa in self.pa(x)}
        # TODO inefficient
        confs = {('>', v) for v in self.V if self.is_confounded(v, x)}
        visited = chs | pas | confs  # this is irrelevant block...
        queue = deque(visited)

        while queue:
            direction, at = queue.popleft()
            if at == y:
                return False

            nexts = set()

            blocked = at in zs
            chs = {('>', ch) for ch in self.ch(at)}
            pas = {('<', pa) for pa in self.pa(at)}
            # TODO inefficient
            confs = {('>', v) for v in self.V if self.is_confounded(v, at)}

            if direction == '>':
                if not blocked:  # --> at --> ch
                    nexts |= chs
                    if at in colliderables:
                        nexts |= pas
                else:  # --> at <-- pa
                    nexts |= pas
                    nexts |= confs  # --> at <----> conf
            else:
                if not blocked:  # <-- at <-- pa, <-- at --> ch, <-- at <----> conf
                    nexts |= pas
                    nexts |= chs
                    nexts |= confs

            for new_dir, new_at in nexts - visited:
                if new_at == y:
                    return False
                visited.add((new_dir, new_at))
                queue.append((new_dir, new_at))

        return True

    root_set = sinks

    def renamed(self, v2v: Dict[str, str], *sets) -> Union['CausalDiagram', Tuple['CausalDiagram', Any]]:
        """ Renamed causal diagram (along with other sets if given) as specified by the dict """

        def rename(v):
            if v in v2v:
                return v2v[v]
            else:
                return v

        G = CausalDiagram({rename(v) for v in sorted(self.V)},
                          {(rename(x), rename(y)) for x, y in self.edges},
                          {(rename(x), rename(y), rename(u)) for x, y, u in self.biedges3})
        if not sets:
            return G
        else:
            return G, *[{rename(v) for v in a_set} for a_set in sets]

    # TODO cache
    def independent(self, xs: VVs, ys: VVs, zs: VVs = frozenset(),
                    verbose=False):
        """ Check whether `xs` and `ys` are independent given `zs` """
        xs, ys, zs = _wrap(xs), _wrap(ys), _wrap(zs)
        assert xs | ys | zs <= self.V, f'unknowns: {(xs | ys | zs) - self.V}'
        xs -= zs
        ys -= zs
        if xs & ys:
            if verbose:
                print(f'{set(xs)} not _||_ {set(ys)} | {set(zs)}')
            return False
        outcome = all(self.__independent(x, y, frozenset(zs)) for x in xs for y in ys)
        if verbose:
            if outcome:
                print(f'{set(xs)} _||_ {set(ys)} | {set(zs)}')
            else:
                print(f'{set(xs)} not _||_ {set(ys)} | {set(zs)}')
        return outcome

    def __repr__(self):
        # TODO the names of unobserved confounders
        return str(self)

    def __str__(self):
        paths, bipaths = _paths_and_bipaths(self)

        modified = True
        while modified:
            modified = False
            for i, path1 in enumerate(bipaths):
                for j, path2 in enumerate(bipaths[i + 1:], i + 1):
                    if path1[-1] == path2[0]:
                        newpath = path1 + path2[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[0] == path2[-1]:
                        newpath = path2 + path1[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[0] == path2[0]:
                        newpath = list(reversed(path2)) + path1[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[-1] == path2[-1]:
                        newpath = path2 + list(reversed(path1))[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                modified = path1 != bipaths[i]
                if modified:
                    break

        # a -> b -> c
        # e -> d -> c
        # == a->b->c<-d<-e
        paths_string = [f'{RIGHT_ARROW}'.join(path) for path in paths]
        bipaths_string = [f'{LR_ARROW}'.join(path) for path in bipaths]
        alone = self.V - {x for path in paths for x in path} - {x for path in bipaths for x in path}

        return f"CausalDiagram('{', '.join([str(x) for x in alone] + paths_string + bipaths_string)}')"

    def dependent(self, xs: VVs, ys: VVs, zs: VVs = frozenset(), verbose=False):
        return not self.independent(xs, ys, zs, verbose)

    def __to_nx(self) -> nx.DiGraph:
        """ Returns a networkx `DiGraph` treating UCs as nodes """
        nxdg = nx.DiGraph()
        nxdg.add_nodes_from(self.V | self.U)
        two_edges = [[(u, x), (u, y)] for x, y, u in self.biedges3]
        nxdg.add_edges_from(itertools.chain(*two_edges))
        nxdg.add_edges_from(self.edges)
        return nxdg

    def draw(self, *,
             prog='neato',
             no_figure=False,
             pos=None,
             xlim=None,
             ylim=None,
             draw_networkx_nodes_kwargs=None,
             draw_networkx_edges_kwargs=None,
             node_callback=None,
             edge_callback=None,
             biedge_callback=None,
             figure_kwargs=None,
             shared=None):
        """ Experimental draw functionality """
        # basic drawing
        if shared is None:
            shared = dict()
        draw_networkx_nodes_kwargs = with_default(draw_networkx_nodes_kwargs, {'node_size': 550, 'node_color': '#ffffff', 'edgecolors': '#000000'})
        draw_networkx_edges_kwargs = with_default(draw_networkx_edges_kwargs, {'arrowsize': 20, 'node_size': 550})
        import matplotlib.pyplot as plt
        if not no_figure:
            if figure_kwargs is not None:
                plt.figure(**figure_kwargs)
            else:
                plt.figure()
        nxdg = nx.DiGraph()
        nxdg.add_nodes_from(self.V)
        nxdg.add_edges_from(self.edges)

        if pos is None and 'pos' in shared:
            pos = shared['pos']

        if pos is None:
            pos = nx.nx_pydot.pydot_layout(nxdg, prog=prog)  # causal order only

        node_collection = nx.draw_networkx_nodes(nxdg, pos, nodelist=(node_list := sorted(nxdg.nodes)), **draw_networkx_nodes_kwargs)
        if node_callback is not None:
            node_callback(node_list, node_collection)
        arcs = nx.draw_networkx_edges(nxdg, pos, edgelist=(edge_list := list(nxdg.edges)), **draw_networkx_edges_kwargs)  # type: List[FancyArrowPatch]
        if edge_callback is not None:
            edge_callback(edge_list, arcs)

        nxbg = nx.DiGraph()
        nxbg.add_nodes_from(self.V)

        nxbg.add_edges_from(bidirected_ordered := list(as_sortups(self.u2vv.values())))
        # noinspection PyTypeChecker
        rads = _draw_helper(pos, self.edges, bidirected_ordered)
        if 'rads' in shared:
            rads = [shared['rads'][k] if k in shared['rads'] else rads[i] for i, k in enumerate(bidirected_ordered)]

        for rad, (va, vb) in zip(rads, bidirected_ordered):
            nx.draw_networkx_edges(nxbg,
                                   pos,
                                   edgelist=[(va, vb)],
                                   style=(0, (8, 8)),
                                   node_size=550,
                                   arrowsize=20,
                                   arrowstyle='<|-|>',
                                   arrows=True,
                                   connectionstyle=ConnectionStyle.Arc3(rad=rad))

        if biedge_callback is not None:
            arcs = nx.draw_networkx_edges(nxbg,
                                          pos,
                                          edgelist=bidirected_ordered,
                                          style=(0, (8, 8)),
                                          node_size=550,
                                          arrowsize=20,
                                          arrowstyle='<|-|>',
                                          arrows=True,
                                          connectionstyle=ConnectionStyle.Arc3(rad=0.4))
            biedge_callback(bidirected_ordered, arcs)

        if not isinstance(draw_networkx_nodes_kwargs['edgecolors'], str):
            assert len(draw_networkx_nodes_kwargs['edgecolors']) == len(node_list)
            for v, c in zip(node_list, draw_networkx_nodes_kwargs['edgecolors']):
                nx.draw_networkx_labels(nxbg, pos, labels={v: v}, font_color=c)
        else:
            nx.draw_networkx_labels(nxbg, pos, font_color=draw_networkx_nodes_kwargs['edgecolors'])

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # override
        if shared is not None:
            if 'xlim' in shared:
                plt.xlim(*shared['xlim'])
            if 'ylim' in shared:
                plt.ylim(*shared['ylim'])

        return {'pos': pos, 'xlim': plt.xlim(), 'ylim': plt.ylim(), 'rads': dict(zip(bidirected_ordered, rads))}

    def fast_latent_projection(self, to_keep: Union[Collection[str], KeysView[str]]) -> 'CausalDiagram':
        to_keep = frozenset(to_keep)
        if to_keep == self.V:
            return self
        assert to_keep <= self.V, f'out of scope {to_keep - self.V}'
        to_keep &= self.V  # if ... no assertion mode

        # as subgraph
        directed_edges = [(W, c) for W in to_keep for c in self._ch[W] & to_keep]
        # not directedly connected (ancestors or descendants)
        for W in to_keep:
            queue = visited_queue(self._ch[W] - to_keep)
            while queue:
                x = queue.pop()
                if x in to_keep:
                    directed_edges.append((W, x))
                else:
                    queue.push_all(self._ch[x])

        bidirected_edges = []  # [(x, y, u) for x, y, u in self.biedges3 if {x, y} <= to_keep]

        _offset = [-1]

        def next_U():
            _offset[0] = _offset[0] + 1
            while f'U_{_offset[0]}' in self.U | self.V:
                _offset[0] = _offset[0] + 1
            return f'U_{_offset[0]}'

        new_an = defaultdict(set)
        for W in to_keep:
            new_an[W].add(W)
            queue = visited_queue(self._pa[W] - to_keep)
            while queue:
                p = queue.pop()
                new_an[W].add(p)
                queue.push_all(self._pa[p] - to_keep)

        for W1, W2 in pairs(to_keep):
            if new_an[W1] & new_an[W2]:
                bidirected_edges.append((W1, W2, next_U()))
            else:
                for a1, a2 in product(new_an[W1], new_an[W2]):
                    assert a1 != a2
                    if self.is_confounded(a1, a2):
                        bidirected_edges.append((W1, W2, next_U()))
        return CausalDiagram(to_keep, directed_edges, bidirected_edges)

    def latent_projection(self, to_keep: Union[Collection[str], KeysView[str]]) -> 'CausalDiagram':
        """ Latent projection onto the given argument """
        # TODO efficiency
        to_keep = frozenset(to_keep)
        if to_keep == self.V:
            return self
        assert to_keep <= self.V, f'out of scope {to_keep - self.V}'
        to_keep &= self.V  # if ... no assertion mode

        dag = self.__to_nx()
        directed_edges = set()
        bidirected_edges = set()

        ancestors = dict()
        for x in dag.nodes:
            ancestors[x] = frozenset(nx.ancestors(dag, x))

        for x, y in itertools.combinations(shuffled(list(to_keep)), 2):
            # x->y
            if dag.has_edge(x, y):
                directed_edges.add((x, y))
            else:
                # x--hidden-->y through
                for s_path in nx.all_simple_paths(dag, x, y):
                    assert len(s_path) > 2
                    if all(v not in to_keep for v in s_path[1:-1]):
                        directed_edges.add((x, y))
                        break

            # y->x
            if dag.has_edge(y, x):
                directed_edges.add((y, x))
            else:
                # y--hidden-->x through
                for s_path in nx.all_simple_paths(dag, y, x):
                    assert len(s_path) > 2
                    if all(v not in to_keep for v in s_path[1:-1]):
                        directed_edges.add((y, x))
                        break
            # x<->y
            for common_ancestor in ancestors[x] & ancestors[y]:
                if common_ancestor not in to_keep:
                    # x<--------hidden
                    for s_pathx in nx.all_simple_paths(dag, common_ancestor, x):
                        if all(v not in to_keep for v in s_pathx[1:-1]):
                            break
                    else:
                        continue
                    # hidden------->y
                    for s_pathy in nx.all_simple_paths(dag, common_ancestor, y):
                        if all(v not in to_keep for v in s_pathy[1:-1]):
                            break
                    else:
                        continue
                    bidirected_edges.add(frozenset({x, y}))
                    break

        offset = 0
        bi3edges = list()
        for (x, y) in bidirected_edges:
            while f'U_{offset}' in to_keep:
                offset += 1
            bi3edges.append((x, y, f'U_{offset}'))
            offset += 1

        return CausalDiagram(to_keep, directed_edges, bi3edges)

    proj = latent_projection
    LP = latent_projection

    def __marginalize_out(self, W: str) -> 'CausalDiagram':
        # TODO faster
        pas = self.pa(W)
        chs = self.ch(W)
        spouses = self.confounded_withs(W)

        keep_edges = {(x, y) for x, y in self.edges if x != W and y != W}
        keep_edges |= set(product(pas, chs))

        keep_biedges = {(x, y, u) for x, y, u in self.biedges3 if x != W and y != W}
        for s, c in itertools.chain(product(spouses, chs), product(chs, chs)):
            if s == c:
                continue
            if not self.is_confounded(s, c):
                assert f'U_{s}{c}' not in self.U
                keep_biedges.add((s, c, f'U_{s}{c}'))

        return CausalDiagram(self.V - {W}, keep_edges, keep_biedges)

    def marginalize_out(self, v_or_vs: VVs) -> 'CausalDiagram':
        """ With projection out the given argument """
        vs = _wrap(v_or_vs)
        assert vs <= self.V, f'out of scope {vs - self.V}'
        vs &= self.V  # if ... no assertion mode

        H = self
        for W in sorted(vs):
            H = H.__marginalize_out(W)
        return H


CD = CausalDiagram


def _uniform_P_U(_):
    return 1.


class StructuralCausalModel:
    def __init__(self, G: CausalDiagram,
                 F: Union[str, Dict[str, Callable[[Dict[str, Any]], Any]]],
                 P_U: Callable[[Dict[str, Any]], float] = None,
                 D: Dict[str, Tuple] = None,
                 more_U: ASet = None,
                 more_v_to_us: Dict[str, ASet] = None,
                 *,
                 no_cache=False,
                 stochastic=False
                 ):
        """
        Structural Causal Model based on semi-Markovian causal graph.

        G: CausalDiagram
            Causal Graph
        F:
            Deterministic functions for each variable where each function takes other variables' values as a `dict`
            Expressions
        P_U:
            (relative) mass function for the joint distribution over P(U).
            It takes all U values as a `dict`, and returns a relative mass.
            For example returning 1 for all cases constitutes a uniform distribution.
        D:
            Domains for UCs and variables (including `more_U` below) represented as
            a dict with variables as the dict's keys and tuples of domain values as the dict's values
        more_U
            unobserved variables specific to endogenous variables (for randomness)
        more_v_to_us
            unobserved variables for each endogenous variable
        """
        # Domain value cannot be none ...
        more_U_from_parse = frozenset()
        self.G = G
        if not isinstance(F, str):
            self.F = F
        else:
            self.F, parse_edges = parse_scm_functions(F, return_edges=True)
            more_U_from_parse = {v for x, y in parse_edges for v in [x, y] if v not in G.V and v not in G.U}

        self.P_U = P_U if P_U is not None else _uniform_P_U
        self.D = with_default(D, defaultdict(lambda: (0, 1)))  # default binary behavior
        # to cover unobserved variables that are not explicit in causal graph
        self.more_U = more_U_from_parse | (frozenset() if more_U is None else frozenset(more_U))
        self.more_v_to_us = more_v_to_us
        if not no_cache:
            self.query00 = functools.lru_cache(1024)(self.query00)
            self.query11 = functools.lru_cache(1024)(self.query11)
        self.stochastic = stochastic

    def values_of(self, vars_: Sequence[str]):
        """ All possible values as tuples of the given sequence of variables """
        return product(*[self.D[V] for V in vars_])

    def query(self, outcome: Union[str, Tuple[str, ...]], condition: dict = None, intervention: dict = None, verbose=False, *, fast=False) -> dict:
        """ A probability distribution of an expression P(Outcome | condition, do(intervention)) where each probability can be accessed by indexing,
        e.g., if outcome={'Y1', 'Y2'} and want to know P(Y1=0, Y2=0| ...), then query(...)[(0, 1)] will give you the probability. """
        if isinstance(outcome, str):
            outcome = (outcome,)
        if condition is None:
            condition = dict()
        if intervention is None:
            intervention = dict()

        new_condition = tuple(sorted([(x, y) for x, y in condition.items()]))
        new_intervention = tuple(sorted([(x, y) for x, y in intervention.items()]))
        # TODO reordered outcome, and reorder
        # TODO experimental feature
        if fast:
            return self.query11(outcome, new_condition, new_intervention, verbose)
        return self.query00(outcome, new_condition, new_intervention, verbose)

    def query00(self, outcome: Tuple, condition: Tuple, intervention: Tuple, verbose=False) -> Dict[Tuple[Any, ...], float]:
        """

        Notes:
            Inefficient since all U values are generated and V values are evaluated entirely.
        """
        condition = dict(condition)
        intervention = dict(intervention)
        if not outcome:
            # condition does not match intervention
            for cv, cval in condition.items():
                if cv in intervention:
                    if cval != intervention[cv]:
                        return defaultdict(lambda: 0.)
            return {tuple(): 1.}

        prob_outcome = defaultdict(lambda: 0)

        U = list(sorted(self.G.U | self.more_U))
        D = self.D
        P_U = self.P_U
        V_ordered = self.G.causal_order()
        if verbose:
            print(f"ORDER: {V_ordered}")
        normalizer = 0

        for u in product(*[D[U_i] for U_i in U]):  # d^|U|
            assigned = dict(zip(U, u))
            p_u = P_U(assigned) if P_U else 1.
            if p_u == 0:
                continue

            # evaluate values
            for V_i in V_ordered:
                if V_i in intervention:
                    assigned[V_i] = intervention[V_i]
                else:
                    assigned[V_i] = self.F[V_i](assigned)  # pa_i including unobserved

            if not all(assigned[V_i] == condition[V_i] for V_i in condition):
                continue
            normalizer += p_u
            prob_outcome[tuple(assigned[V_i] for V_i in outcome)] += p_u

        if prob_outcome:
            # normalize by prob condition
            return defaultdict(lambda: 0, {k: v / normalizer for k, v in prob_outcome.items()})
        else:
            return defaultdict(lambda: np.nan)  # nan or 0?

    # noinspection PyUnusedLocal
    def query11(self, outcome: Tuple, condition: Tuple, intervention: Tuple, verbose=False) -> Dict[Tuple[Any, ...], float]:
        """

        Notes
            Relatively efficient implementation of query.
            When different U values are given, it checks the difference from the previous evaluation, and compute only needed.
            (still experimental)
        """
        # EXPERIMENTAL
        condition = dict(condition)
        intervention = dict(intervention)
        if not outcome:
            # condition does not match intervention
            for cv, cval in condition.items():
                if cv in intervention:
                    if cval != intervention[cv]:
                        return defaultdict(lambda: 0.)
            return {tuple(): 1.}

        # running Rule 2 to move as much condition to intervention is preferred.
        H = self.G.do(intervention.keys())
        V_ordered = sort_with(H.An(set(outcome) | condition.keys()), H.causal_order())  # all required variables ordered.
        # order U from those affecting top to bottom
        U_ordered = list()
        u_to_v = defaultdict(set)  # those directly pointed by exogenous variables
        for v in V_ordered:
            if v in intervention.keys():
                continue
            for u in self.G.UCs(v) | self.more_v_to_us[v]:
                u_to_v[u].add(v)
                if u not in U_ordered:
                    U_ordered.append(u)

        # what variables should be re-evaluated under the change of u, in what order
        # intervention is not included.
        u_to_v_ordered = {u: sort_with(H.De(u_to_v[u]), V_ordered) for u, vs in list(u_to_v.items())}

        u_to_v_ordered_accu = defaultdict(set)
        for i, U_i in enumerate(U_ordered):
            for U_j in U_ordered[i:]:  # inclusive
                u_to_v_ordered_accu[U_i] |= set(u_to_v_ordered[U_j])
        u_to_v_ordered_accu = {u: sort_with(vs, V_ordered) for u, vs in u_to_v_ordered_accu.items()}

        # fixed
        assigned = dict(intervention)
        D = self.D
        P_U = self.P_U
        normalizer = 0
        prob_outcome = defaultdict(lambda: 0)
        last_u = tuple([None])  # remember the last u setting

        for u in product(*[D[U_i] for U_i in U_ordered]):  # d^|U|
            first_changed_u = U_ordered[first_diff_index(last_u, u)]  # got IndexError: list index out of range, why? TODO
            last_u = u  # reuse

            # reuse, override, (actually only overriding from first_changed_u to the end...
            for U_i, u_i in zip(U_ordered, u):
                assigned[U_i] = u_i

            p_u = P_U(assigned) if P_U else 1.
            if p_u == 0:
                continue
            # evaluate values only if affected by changes later than (or equal to) uth_change
            for V_i in u_to_v_ordered_accu[first_changed_u]:
                # V_i in intervention should not appear here ... always override only changed ...
                assigned[V_i] = self.F[V_i](assigned)  # pa_i including unobserved

            if not all(assigned[V_i] == condition[V_i] for V_i in condition):
                continue
            normalizer += p_u
            prob_outcome[tuple(assigned[V_i] for V_i in outcome)] += p_u

        if prob_outcome:
            # normalize by prob condition
            return defaultdict(lambda: 0, {k: v / normalizer for k, v in prob_outcome.items()})
        else:
            return defaultdict(lambda: np.nan)  # nan or 0?

    def sample(self,
               size: int,
               intervention: Union[Tuple[Tuple[str, Any], ...], Dict] = None,
               user_order=None) -> np.ndarray:
        """ Random sample of the given size. Returns a numpy array with variables ordered based on `user_order` if specified or otherwise on the topological order. """
        if self.stochastic:
            return self.stochastic_sample(size, intervention, user_order)

        if intervention is None:
            intervention = dict()  # type: Dict[str, Any]
        elif isinstance(intervention, tuple):
            intervention = dict(intervention)  # type: Dict[str, Any]

        U = list(sorted(self.G.U | self.more_U))
        D = self.D
        P_U = self.P_U
        eval_order = self.G.causal_order()
        V_ordered = with_default(user_order, self.G.causal_order())

        # TODO save time here?
        u_vals = {tuple(u): P_U(dict(zip(U, u))) for u in product(*[D[U_i] for U_i in U])}
        u_configs = tuple(u_vals.keys())  # order preserved! 3.7!
        unit_counts = np.random.multinomial(size, list(u_vals.values()))

        data = np.zeros((size, len(V_ordered)))
        offset = 0
        for ith_unit, how_many in enumerate(unit_counts):
            if how_many == 0:
                continue
            u = u_configs[ith_unit]
            assigned = dict(zip(U, u))
            for V_i in eval_order:
                if V_i in intervention:
                    assigned[V_i] = intervention[V_i]
                else:
                    assigned[V_i] = self.F[V_i](assigned)
            generated = np.array([assigned[V_i] for V_i in V_ordered])
            data[offset:(offset + how_many), :] = generated
            offset += how_many
        np.random.shuffle(data)
        return data

    # Some of U are not entirely controlled by SCM. We sample U ...
    def stochastic_sample(self,
                          size: int,
                          intervention: Union[Tuple[Tuple[str, Any], ...], Dict] = None,
                          user_order=None) -> np.ndarray:
        """ Random sample of the given size. Returns a numpy array with variables ordered based on `user_order` if specified or otherwise on the topological order. """
        if intervention is None:
            intervention = dict()  # type: Dict[str,Any]
        elif isinstance(intervention, tuple):
            intervention = dict(intervention)  # type: Dict[str,Any]

        U = list(sorted(self.G.U | self.more_U))
        D = self.D
        P_U = self.P_U
        eval_order = self.G.causal_order()
        V_ordered = with_default(user_order, self.G.causal_order())

        u_vals = {tuple(u): P_U(dict(zip(U, u))) for u in product(*[D[U_i] for U_i in U])}
        p_us = list(u_vals.values())
        u_config = list(u_vals.keys())

        data = np.zeros((size, len(V_ordered)))
        u_sampleds = np.random.choice(len(p_us), size, p=p_us)
        for ith_iter in range(size):
            assigned = dict(zip(U, u_config[u_sampleds[ith_iter]]))
            for V_i in eval_order:
                if V_i in intervention:
                    assigned[V_i] = intervention[V_i]
                else:
                    assigned[V_i] = self.F[V_i](assigned)
            data[ith_iter, :] = [assigned[V_i] for V_i in V_ordered]

        return data


SCM = StructuralCausalModel


def qcd(paths: Union[str, Collection],
        bidirectedpaths=None,
        u_number_offset=0,
        u_names=None,
        u_name_prefix='U',
        u_name_postfix='',
        Vs=frozenset()) -> CausalDiagram:
    """ Factory method for causal diagram. (i.e., qcd stands for Quick Causal Diagram)

        e.g.,
        qcd(['ABC','BDEF'], ['CADC']) corresponds to the edges in 'A->B->C', 'B->D->E->F', and 'C<->A<->D<->C'.

        paths: str or a collection of sequence of strings (variables)
            If a string is given, it is parsed and the names of UCs are generated following the given UC naming rule
            If a collection is given, each element is treated as a directed path.

        bidirectedpaths: a collection of sequence of strings (variables)
            bidirected paths

        Vs:
            additional observed variables not connected to any other variables.

        Notes
            Guarantees that the bidirected edges will be named following the order.
    """
    if isinstance(paths, str):
        return CausalDiagram(*parse_graph(paths,
                                          u_number_offset=u_number_offset,
                                          u_names=u_names,
                                          u_name_prefix=u_name_prefix,
                                          u_name_postfix=u_name_postfix,
                                          Vs=Vs))

    if bidirectedpaths is None:
        bidirectedpaths = []

    dir_edges = []
    for path in paths:
        for x, y in zip(path, path[1:]):
            dir_edges.append((x, y))

    bidir_edges = []
    if u_number_offset is None:
        assert u_names, 'either numbering or names should be provided'

    if u_names is None:
        u_id = u_number_offset
        for path in bidirectedpaths:
            for x, y in zip(path, path[1:]):
                bidir_edges.append((x, y, u_name_prefix + f'{u_id}' + u_name_postfix))
                u_id += 1
    else:
        u_names_iter = iter(u_names)
        for path in bidirectedpaths:
            for x, y in zip(path, path[1:]):
                bidir_edges.append((x, y, next(u_names_iter)))

    return CausalDiagram(Vs, dir_edges, bidir_edges)


def merge_two_cds(g1: CausalDiagram, g2: CausalDiagram, strict=False) -> CausalDiagram:
    """ Merge two causal diagrams

    Two graphs are merged into one by unionizing vertices and edges.
    Notes:
        No name conflicts allowed.
    """
    # TODO check cycle
    assert g1.U.isdisjoint(g2.V)
    assert g2.U.isdisjoint(g1.V)

    VV = g1.V | g2.V
    EE = set(g1.edges) | set(g2.edges)

    common_UC_names = g1.u2vv.keys() & g2.u2vv.keys()
    common_UCeds = set(g1.u2vv.values()) & set(g2.u2vv.values())

    if all(g1.confounds(u) == g2.confounds(u) for u in common_UC_names) and \
            all(g1.u_of(x, y) == g2.u_of(x, y) for x, y in common_UCeds):
        VWU = set(g1.biedges3) | set(g2.biedges3)
        return CausalDiagram(VV, EE, VWU)

    if strict:
        conflicting_us1 = {u for u in common_UC_names if g1.confounds(u) != g2.confounds(u)}
        conflicting_us2 = {sortup((x, y)) for x, y in common_UCeds if g1.u_of(x, y) != g2.u_of(x, y)}
        raise ValueError(f'conflicting UCs: {conflicting_us1} / {conflicting_us2}')

    confoundeds = set(g1.u2vv.values()) | set(g2.u2vv.values())
    VWU = set()
    to_name = []
    for i, (x, y) in enumerate(confoundeds):
        if g1.is_confounded(x, y):
            VWU.add((x, y, g1.u_of(x, y)))
        else:
            if (u := g2.u_of(x, y)) not in g1.U:
                VWU.add((*sorted((x, y)), u))
            else:
                to_name.append(sortup((x, y)))
    used_u = {u for *_, u in VWU}
    for x, y in to_name:
        if (u := f'U_{x}_{y}') not in used_u:
            VWU.add((x, y, u))
        else:
            raise ValueError('a standard UC name {u} is taken.')

    out = CausalDiagram(VV, EE, VWU)
    out.causal_order()  # to check acyclicity implicitly
    return out


def cd2qcd(G: CausalDiagram, keep_u_names=False) -> str:
    """ Return a python code that can create a CausalDiagram (see also `__str__` in `CausalDiagram`)

    keep_u_names:
        whether to keep the names of the UCs in `G`
    """
    paths, bipaths = _paths_and_bipaths(G)

    if all(len(v) == 1 for path in paths for v in path) and all(len(v) == 1 for path in bipaths for v in path):
        paths = sorted([''.join(path) for path in paths])
        bipaths = sorted([''.join(path) for path in bipaths])

    u_names = []
    for bipath in bipaths:
        for x, y in zip(bipath, bipath[1:]):
            u_names.append(next(iter(G.UCs(x) & G.UCs(y))))

    loners = set()
    for v in G.V:
        if not (G.pa(v) | G.ch(v) | G.UCs(v)):
            loners.add(v)

    after_str = ''
    if keep_u_names:
        after_str += f', u_names={u_names}'
    if loners:
        after_str += f', Vs={loners}'

    return f'qcd({paths}, {bipaths}{after_str})'


def scm(funcs_spec: str, P_U=None, D=None) -> SCM:
    """ Structural Causal Model from the specified function descriptions

    P_U:
        (relative) joint distribution over U
    D:
        domain information of U and V.

    Notes:
        Evaluation can be slower than passing actual functions.
    """
    functors, edges = parse_scm_functions(funcs_spec, return_edges=True)
    Vs = {v for _, v in edges}
    unknowns = set(chain(*edges))

    Us = unknowns - Vs
    directed_edges = {(x, y) for x, y in edges if x in Vs and y in Vs}
    vu_children = sl_group_by(edges, lambda xy: xy[0], transformer=lambda xy: xy[1])
    more_us = set()
    more_v_to_us = defaultdict(list)
    bidirected_edges = []
    for k, kvs in vu_children.items():  # type: str, Sequence[str]
        if k in Us:
            if len(kvs) == 1:
                more_us.add(k)
                more_v_to_us[kvs[0]].append(k)
            else:
                for x, y in pairs(kvs):
                    bidirected_edges.append((x, y, k))

    G = CausalDiagram(Vs, directed_edges, bidirected_edges)
    return StructuralCausalModel(G, functors, P_U=P_U, D=D, more_U=more_us, more_v_to_us=more_v_to_us)  # noqa
