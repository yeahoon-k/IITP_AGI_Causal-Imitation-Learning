import dataclasses
import functools
import itertools
from collections import defaultdict
from typing import FrozenSet, Tuple, Collection, Sequence, Dict, Optional, Union, AbstractSet

import numpy as np
from numpy.random import choice, poisson
from tqdm import trange

from npsem.idty.gid import gID
from npsem.idty.gidpo.exact_cover import ExactCoverSolver
from npsem.idty.gidpo.gidpo_utils import MO, LP
from npsem.idty.prob_eq import Π, Σ, EqNode
from npsem.idty.zid import is_ID
from npsem.model import CausalDiagram, qcd
from npsem.model_utils import random_GYsXsZs
from npsem.parsers import parse_probspec
from npsem.stat_utils import ProbSpec
from npsem.utils import sortup, connected_components, set_ors, pairs, nondominated_sets, fzset_union, bound

VSet = FrozenSet[str]
ASet = AbstractSet[str]
GCFactor = Tuple[VSet, VSet]
Chunk = FrozenSet[int]


@dataclasses.dataclass
class JigsawExample:
    G: CausalDiagram
    query: ProbSpec
    distributions: Sequence[ProbSpec]
    is_identifiable: bool

    def check(self):
        cj = CausalJigsaw(self.query, self.G, JigsawData(self.distributions, self.G.V))  # .refined()
        return cj.is_identifiable() == self.is_identifiable


@functools.lru_cache(1)
def jigsaw_examples() -> Tuple[JigsawExample]:
    def P(first: str, second: Optional[str] = None):
        if second is None:
            return ProbSpec(outcome=list(first))
        else:
            return ProbSpec(intervention=list(first), outcome=list(second))

    examples = []

    G = qcd(['XABY'], [])
    query = ProbSpec(('X',), ('Y',))
    distributions = [P('XB'), P('AY'), P('AB'), P('B', 'Y')]
    examples.append(JigsawExample(G, query, distributions, True))

    # Appendix example
    G = qcd(['XABY'], [])
    query = ProbSpec(('X',), ('Y',))
    distributions = [P('XB'), P('AY'), P('AB')]
    examples.append(JigsawExample(G, query, distributions, False))

    # Figure 1, 7
    G = qcd(['AXBCY', 'AB'], ['XY'])
    query = ProbSpec(('X',), ('Y',))
    distributions = [P('X', 'AC'), P('BCY')]
    examples.append(JigsawExample(G, query, distributions, True))

    # Figure 6
    G = qcd(['AY', 'BY', 'XY'], ['BAX'])
    query = ProbSpec(('X',), ('Y',))
    distributions = [P('AXY'), P('BXY')]
    examples.append(JigsawExample(G, query, distributions, False))

    return tuple(examples)


class JigsawData:
    """ Class representing available partially-observed distributions  """

    def __init__(self, distributions: Union['JigsawData', Sequence[Union[str, ProbSpec]]], Vs=None):
        distributions = [parse_probspec(dist) if isinstance(dist, str) else dist for dist in distributions]

        zv_pairs = [(dist.Xs, dist.Ys) for dist in distributions]
        assert all(not dist.Zs for dist in distributions)
        assert all(a.isdisjoint(b) for a, b in zv_pairs)
        if Vs is not None:
            assert all(a | b <= Vs for a, b in zv_pairs)
        self.zv_pairs = zv_pairs
        self.Zss = tuple(Zs for Zs, _ in self.zv_pairs)
        self.Vss = tuple(Vs_ for _, Vs_ in self.zv_pairs)
        self.distributions = tuple(distributions)

    def with_indices_removed(self, index_or_indices: Union[int, Collection[int]]):
        if isinstance(index_or_indices, int):
            i = index_or_indices
            return JigsawData(self.distributions[:i] + self.distributions[i + 1:])
        else:
            keep = set(range(len(self.zv_pairs))) - set(index_or_indices)
            return JigsawData(tuple(self.distributions[i] for i in keep))

    def with_datum_removed(self, factor: GCFactor):
        for i, zv_i in enumerate(self.zv_pairs):
            if zv_i == factor:
                return JigsawData(self.distributions[:zv_i] + self.distributions[zv_i + 1:])  # noqa
        return self

    def to_probspecs(self):
        return [ProbSpec(xs, ys) for xs, ys in self]

    def __len__(self):
        return len(self.zv_pairs)

    def __iter__(self):
        return iter(self.zv_pairs)

    def refined(self, Vs: ASet) -> 'JigsawData':
        zv_pairs = sortup(frozenset(ProbSpec(Zs_ & Vs, Vs_ & Vs) for Zs_, Vs_ in self.zv_pairs))
        return JigsawData(zv_pairs, Vs)

    def __add__(self, new_data: 'JigsawData'):
        assert NotImplementedError()
        pass

    def __str__(self):
        return str(self.to_probspecs())

    def __repr__(self):
        return str(self.to_probspecs())


class CausalJigsaw:
    __SENTINEL = list('sentinel')

    def __init__(self, query: Union[ProbSpec, str], G: CausalDiagram, data: Union[Sequence[Union[ProbSpec, str]], JigsawData] = None):
        # data-unrelated
        if isinstance(query, str):
            query = parse_probspec(query)
        if data is not None and not isinstance(data, JigsawData):
            data = JigsawData([parse_probspec(dist) if isinstance(dist, str) else dist for dist in data], G.V)

        self.query = query
        self.Xs = frozenset(query.Xs)
        self.Ys = frozenset(query.Ys)
        assert self.Xs.isdisjoint(self.Ys)
        self.G = G

        self.Ys_star = self.Ys
        self.Xs_star = self.G.do(self.Xs).An(self.Ys) & self.Xs
        self.Vs_star = self.Ys_star | self.Xs_star

        self.Ys_plus = self.G.underbar(self.Xs_star).An(self.Ys)
        self.Xs_plus = self.G.An(self.Xs_star) - self.Ys_plus
        self.Vs_plus = self.Ys_plus | self.Xs_plus
        self.G_plus = self.G[self.Vs_plus]

        self.__gc_factors_cache__ = dict()
        self.MVEF = functools.lru_cache()(self.MVEF)
        self.embedding_factor_map = functools.lru_cache()(self.embedding_factor_map)

        # data-related
        self.data = data
        self.chart = JigsawChart(self) if data is not None else None
        self.puzzle = None
        self.output = CausalJigsaw.__SENTINEL

    def with_data_replaced(self, new_data: JigsawData) -> 'CausalJigsaw':
        if not isinstance(new_data, JigsawData):
            new_data = JigsawData(new_data)
        if not self.data:
            return CausalJigsaw(self.query, self.G, new_data)

        # make use of existing chart
        cj2 = CausalJigsaw(self.query, self.G)

        # use cached functions
        cj2.__gc_factors_cache__ = self.__gc_factors_cache__
        cj2.MVEF = self.MVEF
        cj2.embedding_factor_map = self.embedding_factor_map

        # initialize data and chart using existing information
        cj2.data = new_data
        cj2.chart = JigsawChart(cj2, reference_chart=self.chart)
        return cj2

    def is_identifiable(self):
        assert self.data is not None, f'no data is provided'
        if self.output is CausalJigsaw.__SENTINEL:
            self.puzzle = self.chart.to_exact_cover()
            self.output = self.puzzle.solve()
            return self.output is not None
        else:
            return self.output

    def identify(self) -> Optional[EqNode]:
        if not self.is_identifiable():
            return None
        else:
            reverse_lookup = defaultdict(list)
            for data in self.chart.mos_per_data:
                for info in self.chart.mos_per_data[data]:  # info = mos, (lXs, lYs)
                    reverse_lookup[info].append(data)

            eqs = []
            to_margs = self.Ys_plus - self.Ys_star
            for subset in self.output:  #
                with_mo = next(iter(nondominated_sets(self.chart.subset_to_mo[subset])))  # any mo is okay!
                lYs = fzset_union(ps.Ys for ps in subset) - with_mo
                lXs = fzset_union(ps.Xs for ps in subset) - with_mo - lYs

                data_to_use = reverse_lookup[(with_mo, (lXs, lYs))]
                Zs_, Vs_ = next(iter(sorted(data_to_use, key=lambda zv: (len(zv[1]), -len(zv[0])))))  # small Vs_, largest Zs_, (any data is okay!)
                eq = gID(LP(self.G, (Vs_ | Zs_) & self.G.V), lYs, lXs, [Zs_], no_thicket=True)

                to_margs -= with_mo
                eqs.append(eq)

            return Σ(to_margs, Π(eqs))

    def print(self):
        print(f'Query: {self.query}')
        print(f'Graph: {self.G}')
        if self.chart:
            self.chart.print()

    def purge(self) -> 'CausalJigsaw':
        # remove not helpful data ...
        # remove redundants while keeping only one ...
        to_remove = set(self.chart.useless_data() + tuple(itertools.chain(*[red[1:] for red in self.chart.redundants()])))
        simple_data = self.data.with_indices_removed({self.data.zv_pairs.index(f) for f in to_remove})
        assert simple_data is not None
        return self.with_data_replaced(simple_data)

    def refined(self) -> 'CausalJigsaw':
        Xs = self.Xs_star
        G = self.G[self.Vs_plus]
        if self.data:
            return CausalJigsaw(ProbSpec(Xs, self.Ys), G, self.data.refined(self.Vs_plus))
        else:
            return CausalJigsaw(ProbSpec(Xs, self.Ys), G)

    def gc_factors(self, mo: VSet = frozenset()) -> Sequence[Tuple[VSet, VSet]]:
        mo = frozenset(mo)
        if mo in self.__gc_factors_cache__:
            return self.__gc_factors_cache__[mo]
        out = self.__gc_factors(mo)
        self.__gc_factors_cache__[mo] = out
        return out

    def __gc_factors(self, mo: VSet = frozenset()) -> Sequence[Tuple[VSet, VSet]]:
        """ gc-factors given `mo` marginalized out """
        assert mo.isdisjoint(self.Vs_star), 'essentials cannot be marginalized out.'
        assert mo <= self.Ys_plus, 'only from Y+'

        H = MO(self.G_plus, mo)
        return tuple((H.pa(S_i) - S_i, S_i) for S_i in sorted(H[self.Ys_plus - mo].c_components))

    def embedding_factor_map(self, mo1: VSet, mo2: VSet) -> Dict[GCFactor, GCFactor]:
        """ Find who is my embedding factor? F1 -> F2 """
        assert mo1 <= mo2
        gcfs1 = self.gc_factors(mo1)
        gcfs2 = self.gc_factors(mo2)

        diff = mo2 - mo1
        adjs = [((a, b), (c, d)) for (a, b), (c, d) in pairs(gcfs1) if (a | b) & (c | d) & diff]
        merged = dict()
        for concomp in connected_components(gcfs1, adjs):
            signature = set_ors(y for (x, y) in concomp)
            for e, f in gcfs2:
                if f & signature:
                    for (x, y) in concomp:
                        merged[(x, y)] = (e, f)
                    break
        return merged

    def MVEF(self, Xs_: VSet, Ys_: VSet, Zs_: VSet, Vs_: VSet) -> Optional[Tuple[VSet, GCFactor]]:
        """ An MVEF-admissible set and an MVEF (if exists)

        for a factor P(Ys_|do(Xs_)) and distribution P(Vs_|do(Zs_))
        """
        G, starred = self.G_plus, self.Vs_star
        to_mo = frozenset()
        while True:
            lXs, lYs = self.embedding_factor_map(frozenset(), to_mo)[(Xs_, Ys_)]  # TODO efficient

            if not ((lYs & starred <= Vs_) and
                    (lXs & starred <= Vs_ | Zs_) and
                    (Zs_.isdisjoint((G - lXs).An(lYs)))):
                return None

            if (lYs <= Vs_) and (lXs <= Vs_ | Zs_):
                return to_mo, (lXs, lYs)

            mo1 = (lYs - starred) - Vs_
            mo2 = ((lXs - starred) - (Vs_ | Zs_))

            to_mo |= mo1 | mo2


class JigsawChart:
    def __init__(self, jigsaw: CausalJigsaw, *, reference_chart: 'JigsawChart' = None):
        assert jigsaw.data is not None
        self.jigsaw = jigsaw
        self.row_header = jigsaw.gc_factors()
        self.col_header = jigsaw.data.zv_pairs
        self.table = {r: {c: None for c in self.col_header} for r in self.row_header}  # type: Dict[GCFactor, Dict[GCFactor, Optional[Union[bool,VSet]]]]
        self.mos_per_data = defaultdict(set)  # store pairs, for each data, of a set to marginalized and a gc-factor under the marginalization.
        self.subset_to_mo = None
        self.__fill_in(reference_chart)

    def __fill_in(self, reference_chart: 'JigsawChart' = None):
        # G = self.jigsaw.G_plus  # refined G
        G = self.jigsaw.G  # refined G

        for Xs_, Ys_ in self.jigsaw.gc_factors():
            for Zs_, Vs_ in self.jigsaw.data.zv_pairs:
                if reference_chart and (Zs_, Vs_) in reference_chart.col_header:
                    self.table[(Xs_, Ys_)][(Zs_, Vs_)] = reference_chart.table[(Xs_, Ys_)][(Zs_, Vs_)]
                    continue

                output = self.jigsaw.MVEF(Xs_, Ys_, Zs_, Vs_)
                if output is None:
                    self.table[(Xs_, Ys_)][(Zs_, Vs_)] = False
                    continue

                with_mo, (lXs, lYs) = output
                if is_ID(LP(G - Zs_, Vs_ & G.V), lYs, lXs - Zs_):
                    self.table[(Xs_, Ys_)][(Zs_, Vs_)] = with_mo
                    self.mos_per_data[(Zs_, Vs_)].add((with_mo, (lXs, lYs)))
                else:
                    self.table[(Xs_, Ys_)][(Zs_, Vs_)] = False

    def useless_data(self) -> Tuple[GCFactor, ...]:
        skip_data = set()
        for Zs_, Vs_ in self.jigsaw.data.zv_pairs:
            if all(self.table[(Xs_, Ys_)][(Zs_, Vs_)] is False for Xs_, Ys_ in self.jigsaw.gc_factors()):
                skip_data.add((Zs_, Vs_))

        return tuple(skip_data)

    def print(self):
        skip_data = self.useless_data()

        zv_pairs = list(filter(lambda _: _ not in skip_data, self.jigsaw.data.zv_pairs))

        print()
        rows = [ProbSpec(Xs_, Ys_) for Xs_, Ys_ in self.jigsaw.gc_factors()]
        cols = [ProbSpec(Zs_, Vs_) for Zs_, Vs_ in zv_pairs]
        max_row_id_len = max(len(str(r)) for r in rows)
        max_col = len(str(False))
        for Xs_, Ys_ in self.jigsaw.gc_factors():
            for Zs_, Vs_ in zv_pairs:
                if self.table[(Xs_, Ys_)][(Zs_, Vs_)] is not False:
                    max_col = max(max_col, len(str(set(self.table[(Xs_, Ys_)][(Zs_, Vs_)]))))
        max_col_id_len = max(max_col, max(len(str(c)) for c in cols)) if cols else 0

        print(' ' * max_row_id_len, '|', end='', sep='')
        for Zs_, Vs_ in zv_pairs:
            print(str(ProbSpec(Zs_, Vs_)).rjust(max_col_id_len), ' ', sep='', end='')
        print()
        print(('-' * max_row_id_len) + '+' + ('-' * ((max_col_id_len + 1) * len(zv_pairs))))
        for Xs_, Ys_ in self.jigsaw.gc_factors():
            print(str(ProbSpec(Xs_, Ys_)).rjust(max_row_id_len), '|', sep='', end='')
            for Zs_, Vs_ in zv_pairs:
                if self.table[(Xs_, Ys_)][(Zs_, Vs_)] is False:
                    print(str('').rjust(max_col_id_len), ' ', sep='', end='')
                elif self.table[(Xs_, Ys_)][(Zs_, Vs_)]:
                    print(str(set(self.table[(Xs_, Ys_)][(Zs_, Vs_)])).rjust(max_col_id_len), ' ', sep='', end='')
                else:
                    print('{}'.rjust(max_col_id_len), ' ', sep='', end='')
            print()
        print()

    def to_exact_cover(self) -> ExactCoverSolver:
        elements = tuple(ProbSpec(Xs_, Ys_) for Xs_, Ys_ in self.row_header)
        subsets = set()
        self.subset_to_mo = defaultdict(set)
        for Zs_, Vs_ in self.col_header:
            for Xs_, Ys_ in self.row_header:
                if (with_mo := self.table[(Xs_, Ys_)][(Zs_, Vs_)]) is not False:
                    if with_mo:
                        embedded_factors = frozenset(ProbSpec(lXs_, lYs_) for lXs_, lYs_ in self.row_header if (lXs_ | lYs_) & with_mo)
                    else:
                        embedded_factors = frozenset((ProbSpec(Xs_, Ys_),))

                    subsets.add(embedded_factors)
                    self.subset_to_mo[embedded_factors].add(with_mo)

        return ExactCoverSolver(elements, subsets)

    def redundants(self) -> Tuple[Tuple[GCFactor, ...], ...]:
        info = dict()
        for Zs_, Vs_ in self.col_header:
            info[(Zs_, Vs_)] = tuple(self.table[(Xs_, Ys_)][(Zs_, Vs_)] for Xs_, Ys_ in self.row_header)

        group_by_values = defaultdict(list)
        for k, vs in info.items():
            group_by_values[vs].append(k)

        return tuple(tuple(ks) for vs, ks in group_by_values.items() if len(ks) >= 1)


def fzch(a: Collection[str], size: int) -> FrozenSet[str]:
    return frozenset(choice(list(a), size, replace=False))


def random_jigsaws(total, is_tqdm=True, max_v=7):
    for _ in (trange(total, smoothing=0.01) if is_tqdm else range(total)):
        G, *_ = random_GYsXsZs(max_size=None, Y_rootset=np.random.rand() < 0.8, n_V=min(max_v, 4 + poisson(2.5)))
        Vs = list(G.V)
        for _ in range(len(G.V)):
            Ys = fzch(Vs, bound(1 + poisson(1.0), 1, len(Vs)))
            assert Ys
            if G.V - Ys:
                Xs = fzch(list(G.V - Ys), min(len(G.V - Ys), poisson(1.0)))
            else:
                Xs = frozenset()

            Zss = [fzch(Vs, min(len(Vs), poisson(1.))) for _ in range(poisson(len(G.V) / 2) + 1)]
            # print(*[str(set(Zs)) if Zs else '{}' for Zs in Zss])

            dists = [ProbSpec(Zs,
                              fzch(set(Vs) - Zs, max(1, len(set(Vs) - Zs) - poisson(2.0))))
                     for Zs in Zss if set(Vs) - Zs]

            if not dists:
                continue
            yield CausalJigsaw(ProbSpec(Xs, Ys), G, JigsawData(dists, G.V))  # .refined()
