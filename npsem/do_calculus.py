from collections import deque
from functools import lru_cache
from typing import AbstractSet, Set, FrozenSet

from npsem.model import CausalDiagram
from npsem.stat_utils import ProbSpec
from npsem.utils import nonempties, combinations, as_sets

VSet = FrozenSet[str]


class DoCalculus:
    """ a suite of tools for the rules of do-calculus """

    def __init__(self, G: CausalDiagram):
        self.G = G

    def deletable_observation(self, spec: ProbSpec):
        """ returns a maximal set of observations that can be removed based on Rule 1 """
        GdoX = self.G.do(spec.Xs)
        Zs_ = frozenset()
        for Zs in nonempties(as_sets(combinations(spec.Z))):
            Ws = spec.Zs - Zs
            if GdoX.independent(spec.Ys, Zs, Ws | spec.Xs):
                assert self.is_rule_1_del_obs(spec, Zs)
                Zs_ = Zs
        return Zs_

    def is_rule_1_del_obs(self, spec: ProbSpec, Zs: AbstractSet[str]) -> bool:
        if not Zs:
            return True

        H = self.G - spec.Xs
        Ys, WZs = spec.Ys, spec.Zs
        assert Zs <= WZs
        Ws = WZs - Zs
        return H.independent(Ys, Zs, Ws)

    #
    def is_rule_1_add_obs(self, spec: ProbSpec, Zs: AbstractSet[str]) -> bool:
        if not Zs:
            return True

        H = self.G - spec.Xs
        Ys, Ws = spec.Ys, spec.Zs
        assert Ws.isdisjoint(Zs)
        return H.independent(Ys, Zs, Ws)

    def is_rule_2_del_obs_add_act(self, spec: ProbSpec, Zs: AbstractSet[str]) -> bool:
        if not Zs:
            return True

        H = self.G - spec.Xs
        Ys, WZs = spec.Ys, spec.Zs
        assert Zs <= WZs
        Ws = WZs - Zs
        return H.underbar(Zs).independent(Ys, Zs, Ws)

    def is_rule_2_del_act_add_obs(self, spec: ProbSpec, Zs: AbstractSet[str]) -> bool:
        if not Zs:
            return True

        Ys, Ws = spec.Ys, spec.Zs
        XZs = spec.Xs
        assert Zs <= XZs
        Xs = XZs - Zs
        H = self.G - Xs
        return H.underbar(Zs).independent(Ys, Zs, Ws)

    def is_rule_3_del_act(self, spec: ProbSpec, Zs: AbstractSet[str]) -> bool:
        if not Zs:
            return True

        Ys, Ws = spec.Ys, spec.Zs
        XZs = spec.Xs
        assert Zs <= XZs
        Xs = XZs - Zs
        H = self.G - Xs
        return H.do(Zs - H.An(Ws)).independent(Ys, Zs, Ws)

    def is_rule_3_add_act(self, spec: ProbSpec, Zs: AbstractSet[str]) -> bool:
        if not Zs:
            return True

        Ys, Ws = spec.Ys, spec.Zs
        Xs = spec.Xs
        assert Zs.isdisjoint(Xs)
        H = self.G - Xs
        return H.do(Zs - H.An(Ws)).independent(Ys, Zs, Ws)

    def equivalent(self, spec: ProbSpec) -> Set[ProbSpec]:
        """ a not-so-efficient way to enumerate all probability specifications equivalent to the given specification"""
        specs = {spec}
        queue = deque()
        queue.append(spec)

        while queue:
            pop_spec = queue.pop()  # type:ProbSpec

            for z in pop_spec.Zs:
                if self.is_rule_1_del_obs(pop_spec, {z}):
                    if (new_spec := ProbSpec(pop_spec.Xs, pop_spec.Ys, pop_spec.Zs - {z})) not in specs:
                        specs.add(new_spec)
                        queue.append(new_spec)

            for z in self.G.V - pop_spec.XYZs:
                if self.is_rule_1_add_obs(pop_spec, {z}):
                    if (new_spec := ProbSpec(pop_spec.Xs, pop_spec.Ys, pop_spec.Zs | {z})) not in specs:
                        specs.add(new_spec)
                        queue.append(new_spec)

            for z in pop_spec.Xs:
                if self.is_rule_2_del_act_add_obs(pop_spec, {z}):
                    if (new_spec := ProbSpec(pop_spec.Xs - {z}, pop_spec.Ys, pop_spec.Zs | {z})) not in specs:
                        specs.add(new_spec)
                        queue.append(new_spec)

            for z in pop_spec.Zs:
                if self.is_rule_2_del_obs_add_act(pop_spec, {z}):
                    if (new_spec := ProbSpec(pop_spec.Xs | {z}, pop_spec.Ys, pop_spec.Zs - {z})) not in specs:
                        specs.add(new_spec)
                        queue.append(new_spec)

            for z in pop_spec.Xs:
                if self.is_rule_3_del_act(pop_spec, {z}):
                    if (new_spec := ProbSpec(pop_spec.Xs - {z}, pop_spec.Ys, pop_spec.Zs)) not in specs:
                        specs.add(new_spec)
                        queue.append(new_spec)

            for z in self.G.V - pop_spec.XYZs:
                if self.is_rule_3_add_act(pop_spec, {z}):
                    if (new_spec := ProbSpec(pop_spec.Xs | {z}, pop_spec.Ys, pop_spec.Zs)) not in specs:
                        specs.add(new_spec)
                        queue.append(new_spec)
        return specs

    @lru_cache
    def minimal(self, spec: ProbSpec):
        for Z in set(spec.Zs):
            if self.is_rule_1_del_obs(spec, {Z}):
                spec = spec.with_Z(spec.Zs - {Z})
        for Z in set(spec.Zs):
            if self.is_rule_2_del_obs_add_act(spec, {Z}):
                spec = spec.with_Z(spec.Zs - {Z}).with_X(spec.Xs | {Z})
        for X in set(spec.Xs):
            if self.is_rule_3_del_act(spec, {X}):
                spec = spec.with_X(spec.Xs - {X})
        return spec

    def maximal_Xs(self, spec: ProbSpec):
        for X in set(self.G.V - spec.XYZs):
            if self.is_rule_3_add_act(spec, {X}):
                spec = spec.with_X(spec.Xs | {X})
        return spec
