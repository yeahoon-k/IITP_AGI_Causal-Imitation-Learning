import collections
from collections import defaultdict
from itertools import product
from typing import Iterable, List, Union, AbstractSet, Sequence, Tuple, Collection, FrozenSet, Optional

import numpy as np

from npsem.model import StructuralCausalModel as SCM
from npsem.stat_utils import ProbSpec, IQueryable
from npsem.utils import dict_or, split_by, set_union, mults_zero, sortup, mults, fzset_union, class_split_by

"""
This module deals with formula for identifiability and transportability.
"""


class EqNode:
    """ Node of expression tree """

    def __init__(self):
        self.children = []
        self.defined_over = None

    @property
    def child(self):
        if len(self.children) == 1:
            return self.children[0]
        else:
            return None

    @child.setter
    def child(self, x):
        self.children = [x]

    def evaluate(self, value_assignment: dict, var_domains, zero_for_nan=False):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.str2(defaultdict(lambda x: x.lower())).strip()

    def str2(self, vv) -> str:
        raise NotImplementedError()

    def __eq__(self, other):
        return self is other

    def attach_distribution(self, distr: Union[IQueryable, SCM, Sequence[SCM]]):
        attach_distribution(self, distr)
        return self

    def detach_distribution(self):
        attach_distribution(self, None)

    def updated(self, new_children: List['EqNode']) -> 'EqNode':
        raise NotImplementedError()


class ConstantNode(EqNode):
    """ Constant """

    def __init__(self, val):
        super().__init__()
        self.val = val
        self.defined_over = frozenset()

    def evaluate(self, value_assignment: dict, var_domains, zero_for_nan=False):
        return self.val

    def str(self):
        return str(self.val)

    def str2(self, vv):
        return str(self.val)

    def __eq__(self, other):
        return isinstance(other, ConstantNode) and self.val == other.val

    def inversed(self):
        return ConstantNode(1 / self.val)

    def updated(self, new_children):
        assert not new_children
        return self


ONE1 = ConstantNode(1)


class SumNode(EqNode):
    def __init__(self, sum_over_vars: Collection[str], p_term: EqNode):
        super().__init__()
        assert sum_over_vars
        assert not isinstance(p_term, SumNode)
        self.sum_over_vars = tuple(sorted(sum_over_vars))
        self.children = [p_term]
        assert set(sum_over_vars) <= self.child.defined_over
        self.defined_over = p_term.defined_over - set(sum_over_vars)

    def evaluate(self, value_assignment: dict, var_domains: dict, zero_for_nan=False):
        total = 0.0
        for values in product(*[var_domains[var] for var in self.sum_over_vars]):
            new_values = dict_or(value_assignment, dict(zip(self.sum_over_vars, values)))
            cur_val = self.child.evaluate(new_values, var_domains, zero_for_nan)
            total += cur_val
            if np.isnan(total):
                return np.nan
        return total

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.sum_over_vars:
            under = ', '.join([str(V).lower() for V in self.sum_over_vars])
            return f"\\sum_{{{under}}} " + str(self.child).strip()
        else:
            return str(self.child).strip()

    def __eq__(self, other):
        return isinstance(other, SumNode) and self.sum_over_vars == other.sum_over_vars and self.child == other.child

    def str2(self, vv) -> str:
        if self.sum_over_vars:
            vv = dict(vv)
            for V in self.sum_over_vars:
                if V in vv:
                    vv[V] = vv[V] + "'"
                else:
                    vv[V] = V.lower()

        if self.sum_over_vars:
            under = ', '.join([vv[V] for V in self.sum_over_vars])
            return f"\\sum_{{{under}}} " + self.child.str2(vv).strip()
        else:
            return self.child.str2(vv).strip()

    def updated(self, new_children: List['EqNode']) -> 'EqNode':
        return Σ(self.sum_over_vars, new_children[0])


class ProductNode(EqNode):
    def __init__(self, terms: Iterable[EqNode]):
        super().__init__()
        assert not any(isinstance(term, ProductNode) for term in terms)
        terms = list(terms)
        sums = [t for t in terms if isinstance(t, SumNode)]
        non_sums = [t for t in terms if not isinstance(t, SumNode)]
        self.children = non_sums + sums
        self.defined_over = set_union(set(t.defined_over) for t in terms)

    def evaluate(self, value_assignment: dict, var_domains, zero_for_nan=False):
        return mults_zero(ch.evaluate(value_assignment, var_domains, zero_for_nan) for ch in self.children)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return " ".join(sorted([str(t).strip() for t in self.children]))

    def __eq__(self, other):
        return isinstance(other, ProductNode) and len(self.children) == len(other.children) and sorted(
            [str(ch) for ch in self.children]) == sorted([str(ch) for ch in other.children])

    # noinspection DuplicatedCode
    def str2(self, vv):
        strings = [t.str2(vv).strip() for t in self.children]

        num_sum_terms = sum([isinstance(ch, SumNode) for ch in self.children])
        if num_sum_terms == 0:
            pairs = [(-len(_), _) for _ in strings]
            pairs = sorted(pairs)
            strings = [the_str2 for _, the_str2 in pairs]
            inner = " ".join(strings)
            return inner.strip()
        elif num_sum_terms == 1:
            pairs = [(isinstance(t, SumNode), -len(_), _) for t, _ in zip(self.children, strings)]
            pairs = sorted(pairs)
            strings = [the_str2 for _, _, the_str2 in pairs]
            inner = " ".join(strings)
            return inner.strip()
        else:
            for i, t in enumerate(self.children):
                if isinstance(t, SumNode):
                    strings[i] = '\\left(' + strings[i] + '\\right)'
            pairs = [(isinstance(t, SumNode), -len(_), _) for t, _ in zip(self.children, strings)]
            pairs = sorted(pairs)
            strings = [the_str2 for _, _, the_str2 in pairs]
            inner = " ".join(strings)
            return inner.strip()

    def updated(self, new_children: List['EqNode']) -> 'EqNode':
        return Π(new_children)


class AssignNode(EqNode):
    def __init__(self, inner, assigning: Collection[str]):
        super().__init__()
        assert not isinstance(inner, AssignNode)
        # self.child = inner
        self.children = [inner]
        self.assigning = frozenset(assigning) & frozenset(self.child.defined_over)
        assert self.assigning
        self.defined_over = inner.defined_over - self.assigning

    def evaluate(self, value_assignment: dict, var_domains, zero_for_nan=False):
        # actually taking only the first item from domains...
        # Big Note Here, it is possible that e.g., W->Z->X->Y X<->W<->Y<->Z, Px(Y)=Pz(Y|X)
        # but Pz(X) can be 0, Pz(Y,X)/Pz(X)
        # print("assignment weighted average required")

        # let unassigned to be decided ... later?
        # for V in self.assigning:
        #     value_assignment[V] = np.nan
        # if set(value_assignment.keys()).isdisjoint(self.assigning):
        #     return self.child.evaluate(value_assignment, var_domains, zero_for_nan)
        # else:
        #     return self.child.evaluate(value_assignment, var_domains, zero_for_nan)

        # warnings.warn('need to study more about assign!!')
        local_assigning = tuple(sorted(self.assigning & set(self.child.defined_over)))
        outs = []
        out = np.nan
        for values in product(*[var_domains[var] for var in local_assigning]):
            new_values = dict_or(value_assignment, dict(zip(local_assigning, values)))
            out = self.child.evaluate(new_values, var_domains, zero_for_nan)
            outs.append(out)
            if not np.isnan(out):
                return out
        return out

    def __eq__(self, other):
        return isinstance(other, AssignNode) and self.assigning == other.assigning and self.child == other.child

    def __str__(self):
        return str(self.child)

    def str2(self, vv):
        if self.assigning:
            vv = dict(vv)
            for V in self.assigning:
                if V in vv:
                    vv[V] = vv[V] + "'"
                else:
                    vv[V] = V.lower()

        if self.assigning:
            return '\\left[' + self.child.str2(vv) + f'\\right]_{{{", ".join([vv[_] for _ in self.assigning])}}}'
        else:
            return self.child.str2(vv)

    def updated(self, new_children: List['EqNode']) -> 'EqNode':
        return assign(new_children[0], self.assigning)


class ProbDistr(EqNode):  # "Stub"
    def __init__(self, intervention=frozenset(), vs=None, domain_id=None):
        super().__init__()
        intervention = frozenset(intervention)
        vs = frozenset(vs)
        assert vs is not None

        self.intervention = intervention
        self.intervention_sortup = sortup(self.intervention)
        self.defined_over = vs | intervention  # what ever vs missed intervention or not

        self.domain_id = domain_id
        self.attached_distribution = None

    def __eq__(self, other):
        return isinstance(other,
                          ProbDistr) and other.intervention == self.intervention and other.domain_id == self.domain_id

    def evaluate(self, value_assignment: dict, var_domains, zero_for_nan=False):
        raise AssertionError('hu!')

    def __repr__(self):
        return str(self)

    def __str__(self):
        domain_id_str = ''
        if self.domain_id is not None:
            domain_id_str = f'^{self.domain_id}'
        if self.intervention:
            dodo = ', '.join([str(_).lower() for _ in self.intervention_sortup])
            return f"P{domain_id_str}_{{{dodo}}}"
        else:
            return f"P{domain_id_str}"

    def str2(self, vv):
        domain_id_str = ''
        if self.domain_id is not None:
            domain_id_str = f'^{self.domain_id}'

        if self.intervention:
            for _ in self.intervention:
                if _ not in vv:
                    vv[_] = _.lower()
            dodo = ', '.join([vv[_] for _ in self.intervention_sortup])
            return f" P{domain_id_str}_{{{dodo}}}"
        else:
            return f" P{domain_id_str}"

    def updated(self, new_children: List['EqNode']) -> 'EqNode':
        assert not new_children
        return self


class FracNode(EqNode):
    def __init__(self, top: EqNode, bottom: EqNode):
        super().__init__()
        self.children = [top, bottom]
        # self.top = top
        # self.bottom = bottom
        # self.defined_over = top.defined_over | bottom.defined_over

    @property
    def defined_over(self):
        return self.top.defined_over | self.bottom.defined_over

    @defined_over.setter
    def defined_over(self, value):
        return

    @property
    def top(self):
        return self.children[0]

    @property
    def bottom(self):
        return self.children[1]

    def evaluate(self, value_assignment: dict, var_domains, zero_for_nan=False):
        bottom_eval = self.bottom.evaluate(value_assignment, var_domains, zero_for_nan)
        if bottom_eval == 0:
            return np.nan
        else:
            return self.top.evaluate(value_assignment, var_domains, zero_for_nan) / bottom_eval

    def __eq__(self, other):
        return isinstance(other, FracNode) and self.top == other.top and self.bottom == other.bottom

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '{' + str(self.top) + '} / {' + str(self.bottom) + '}'

    def str2(self, vv):
        return '\\frac{' + self.top.str2(vv) + '}{' + self.bottom.str2(vv) + '}'

    def inversed(self):
        return frac_node(self.bottom, self.top)

    def updated(self, new_children: List['EqNode']) -> 'EqNode':
        return frac_node(*new_children)


class ProbTerm(EqNode):
    def __init__(self, P: ProbDistr, measurement, condition=None):
        super().__init__()

        assert isinstance(P, ProbDistr), 'not sure...'
        self.P = P  # type: ProbDistr
        # self.child = P  # type: ProbDistr
        self.children = [P]  # type: List[ProbDistr]
        self.measurement = tuple(sorted(measurement))
        self.condition = tuple(sorted(condition)) if condition is not None else tuple()
        assert set(self.measurement) | set(self.condition) <= self.P.defined_over
        self.defined_over = set(self.measurement) | set(self.condition) | self.P.intervention
        assert isinstance(self.P, ProbDistr) or not set(self.condition)

    @staticmethod
    def from_probspec(spec: ProbSpec, vs: AbstractSet[str]) -> 'ProbTerm':
        return ProbTerm(ProbDistr(spec.Xs, vs), spec.Ys, spec.Zs)

    def __eq__(self, other):
        return isinstance(other, ProbTerm) and (other.P, other.measurement, other.condition) == (
            self.P, self.measurement, self.condition)

    def evaluate(self, value_assignment: dict, var_domains, zero_for_nan=False):
        if isinstance(self.P.attached_distribution, SCM):
            zs = {var: value_assignment[var] for var in self.condition}
            xs = {do_var: value_assignment[do_var] for do_var in self.P.intervention}

            query_output = self.P.attached_distribution.query(tuple(sorted(self.measurement)),
                                                              condition=zs,
                                                              intervention=xs)
            ret = query_output[tuple(value_assignment[m] for m in tuple(sorted(self.measurement)))]
            if zero_for_nan and ret == 0.0:
                ret = np.nan
            return ret

        elif isinstance(self.P.attached_distribution, collections.Sequence):
            zs = {var: value_assignment[var] for var in self.condition}
            xs = {do_var: value_assignment[do_var] for do_var in self.P.intervention}

            scm = self.P.attached_distribution[self.P.domain_id]
            query_output = scm.query(tuple(sorted(self.measurement)),
                                     condition=zs,
                                     intervention=xs)
            ret = query_output[tuple(value_assignment[m] for m in tuple(sorted(self.measurement)))]
            if zero_for_nan and ret == 0.0:
                ret = np.nan
            return ret
        else:
            ys = tuple((var, value_assignment[var]) for var in self.measurement)  # already sorted!
            zs = tuple((var, value_assignment[var]) for var in self.condition)  # already sorted!
            xs = tuple((do_var, value_assignment[do_var]) for do_var in self.P.intervention_sortup)
            return self.P.attached_distribution.query(ys, condition=zs, intervention=xs)

    def str2(self, vv):
        for _ in set(self.measurement) | set(self.condition):
            if _ not in vv:
                vv[_] = _.lower()

        if isinstance(self.P, ProbDistr):
            if self.condition:
                return self.P.str2(
                    vv) + f"({','.join([vv[_] for _ in self.measurement])} | {','.join([vv[_] for _ in self.condition])})"
            else:
                return self.P.str2(vv) + f"({','.join([vv[_] for _ in self.measurement])})"
        else:
            return self.P.str2(vv)

    def updated(self, new_children: List['EqNode']) -> 'EqNode':
        return Pr(new_children[0], self.measurement, self.condition)

    def as_probspec(self):
        return ProbSpec(self.P.intervention, self.measurement, self.condition)


def attach_distribution(head: EqNode, distr):
    queue = [head]
    while queue:
        node = queue.pop(0)
        if isinstance(node, ProbDistr):
            node.attached_distribution = distr
        assert (node.child, node.children) != (None, None)
        if node.child is not None:
            queue.append(node.child)
        elif node.children:
            queue.extend(node.children)


def frac_node(top: EqNode, bottom: EqNode) -> EqNode:
    if isinstance(bottom, ConstantNode):
        return Π([top, bottom.inversed()])

    if isinstance(top, ConstantNode) and top != ONE1:
        return Π([top, frac_node(ONE1, bottom)])

    if bottom == ONE1:
        return top

    return FracNode(top, bottom)


def refine_sum_to_1(sum_over_vars: Iterable, terms: Iterable[EqNode]) -> Tuple[FrozenSet, Sequence[EqNode]]:
    """ Removes a term, which will be marginalized out """
    terms = list(terms)
    sum_over_vars = frozenset(sum_over_vars)
    if len(terms) > 1:
        return sum_over_vars, terms

    term = terms[0]
    if isinstance(term, ProbTerm):
        removable = sum_over_vars & frozenset(term.measurement)
        if removable:
            return sum_over_vars - removable, [Pr(term.P, frozenset(term.measurement) - removable, term.condition)]

    return sum_over_vars, terms


# noinspection NonAsciiCharacters
def Σ(sum_over_vars, p_term: EqNode) -> EqNode:
    """ Sum over a probability expression """
    sum_over_vars = frozenset(sum_over_vars)

    if not sum_over_vars:
        return p_term

    # This is the case where passing marginalized distribution, which is not necessary
    if isinstance(p_term, ProbDistr):
        assert p_term.defined_over >= sum_over_vars
        new_p_term = ProbDistr(p_term.intervention - sum_over_vars,
                               p_term.defined_over - sum_over_vars, domain_id=p_term.domain_id)
        return new_p_term

    elif isinstance(p_term, SumNode):
        assert sum_over_vars.isdisjoint(set(p_term.sum_over_vars))
        return Σ(sum_over_vars | set(p_term.sum_over_vars), p_term.child)

    elif isinstance(p_term, ProductNode):
        orglen = len(p_term.children)
        sum_over_vars, remains = refine_sum_to_1(sum_over_vars, p_term.children)
        if not remains:  # Happening????
            return ONE1
        irre, rele = split_by(remains, lambda ch: set(ch.defined_over).isdisjoint(sum_over_vars))
        if irre:
            if rele:
                return Π(irre + [Σ(sum_over_vars, Π(rele))])
            else:  # Happening????
                return Π(irre)
        else:
            if len(remains) != orglen:  # Happening????
                return Σ(sum_over_vars, Π(remains))

    elif isinstance(p_term, ProbTerm):
        sum_over_vars, remains = refine_sum_to_1(sum_over_vars, [p_term])
        if not remains:  # Happening????
            return ONE1
        elif remains[0] != p_term:
            return Σ(sum_over_vars, remains[0])

    return SumNode(sum_over_vars, p_term)


# noinspection DuplicatedCode
def __collapse_prods(terms):
    # P(a,b|c,d)P(c|d) = P(a,b,c|d)
    terms = list(terms)
    while True:
        changed = False

        for i, term1 in enumerate(terms):
            for j, term2 in enumerate(terms[i + 1:], i + 1):
                if isinstance(term1, ProbTerm) and isinstance(term2, ProbTerm) and term1.P == term2.P:
                    # something fish here?
                    if isinstance(term1.P,
                                  ProbDistr) and term1.P.defined_over != term2.P.defined_over and term1.P.domain_id == term2.P.domain_id:
                        new_probdistr = ProbDistr(term1.P.intervention,
                                                  (term1.P.defined_over | term2.P.defined_over) - term1.P.intervention,
                                                  domain_id=term1.P.domain_id)
                    else:
                        new_probdistr = term1.P
                    if set(term2.measurement + term2.condition) == set(term1.condition):
                        new_c = set(term1.condition) & set(term2.condition)
                        new_m = set(term1.measurement) | set(term1.condition) - new_c
                        terms[i] = Pr(new_probdistr, new_m, new_c)
                        terms.pop(j)
                        changed = True
                    elif set(term1.measurement + term1.condition) == set(term2.condition) and term1.P == term2.P:
                        new_c = set(term2.condition) & set(term1.condition)
                        new_m = set(term2.measurement) | set(term2.condition) - new_c
                        terms[i] = Pr(new_probdistr, new_m, new_c)
                        terms.pop(j)
                        changed = True

                if changed:
                    break
            if changed:
                break
        if not changed:
            const_val = 1
            to_pop = []
            for i, t in enumerate(terms):
                if isinstance(t, ConstantNode):
                    const_val *= t.val
                    to_pop.append(i)
            for at in reversed(to_pop):
                terms.pop(at)
            if const_val != 1:
                terms = terms + [ConstantNode(const_val)]
            return terms


# noinspection NonAsciiCharacters
def Π(terms: Iterable[EqNode]) -> EqNode:
    # ignore 1s
    # multiply constants
    # no nested product (but do not check recursively.)
    # collapse terms if possible
    terms = list(terms)
    terms = [t for t in terms if t != ONE1]

    if not terms:
        return ONE1

    # consts, non_consts = split_by(terms, lambda ch: isinstance(ch, ConstantNode))
    consts, non_consts = class_split_by(terms, ConstantNode)
    if len(consts) >= 2:
        merged = ConstantNode(mults(const_node.val for const_node in consts))
        return Π([merged, *non_consts])

    offset = 0
    while True:
        for i, term in enumerate(terms[offset:], offset):
            if isinstance(term, ProductNode):
                terms = list(terms)
                terms.pop(i)
                terms.extend(term.children)
                break
        else:
            break

    if len(terms) > 1:
        terms = __collapse_prods(terms)

    if len(terms) == 1:
        return terms[0]  # next(iter(terms))
    else:
        return ProductNode(terms)


def assign(inner: EqNode, assigning: Collection[str]) -> Optional[EqNode]:
    if inner is None:
        return None
    if set(inner.defined_over) & set(assigning):
        if isinstance(inner, AssignNode):
            return AssignNode(inner.child, set(assigning) | set(inner.assigning))
        else:
            return AssignNode(inner, set(assigning))
    else:
        return inner


def Pr(P: EqNode, measurement: Optional[Collection], condition: Optional[Collection] = None, scope: AbstractSet = None):
    if not measurement:
        return ONE1
    if condition is None:
        condition = set()
    measurement = set(measurement)
    condition = set(condition)
    if scope is None:
        scope = measurement | condition
    assert measurement | condition <= scope
    measurement = set(measurement)
    condition = set(condition)

    if isinstance(P, ProbDistr):
        return ProbTerm(P, measurement, condition)
    elif not condition:
        if measurement == scope:
            return P
        else:
            return Σ(scope - measurement, Pr(P, scope, scope=scope))
    else:
        # Q(a|b) = Q(a,b)/Q(b) = (sum_c' Q(a,b,c'))/(sum_c'a' Q(a,b,c))
        return frac_node(Pr(P, measurement | condition, scope=scope), Pr(P, condition, scope=scope))


def used_dos(root: EqNode) -> FrozenSet[FrozenSet[str]]:
    dos = set()
    if isinstance(root, ProbDistr):
        dos.add(frozenset(root.intervention))
    return fzset_union(used_dos(ch) for ch in root.children) | dos


def used_terms(root: EqNode) -> FrozenSet[ProbSpec]:
    specs = set()
    if isinstance(root, ProbTerm):
        specs.add(ProbSpec(root.P.intervention, root.measurement, root.condition))
    return fzset_union(used_terms(ch) for ch in root.children) | specs


def used_domains(root: EqNode) -> FrozenSet[int]:
    """ Gather domains (multiple environments) used in the formula """
    doms = set()
    if isinstance(root, ProbTerm):
        doms.add(root.P.domain_id)
    return fzset_union(used_domains(ch) for ch in root.children) | doms


def used_vars(root: EqNode) -> FrozenSet[str]:
    return frozenset(set_union(t.XYZs for t in used_terms(root)))


def is_simple_formula(root: EqNode) -> bool:
    return isinstance(root, ProbTerm) or (isinstance(root, AssignNode) and isinstance(root.child, ProbTerm))


def formula_key(formula: EqNode) -> str:
    # TODO is it ... safe?
    return str(formula).replace('  ', ' ').lstrip().rstrip()
