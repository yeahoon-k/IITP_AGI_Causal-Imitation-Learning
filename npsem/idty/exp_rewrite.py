from typing import List, Tuple, Optional

from npsem.idty.prob_eq import EqNode, ProductNode, Π, ConstantNode, FracNode, ONE1, frac_node, ProbTerm, SumNode, Σ, Pr
from npsem.model import CausalDiagram
from npsem.do_calculus import DoCalculus
from npsem.utils import MutableBoolean, split_by, mults, notnone, class_split_by

"""
Methods for rewriting an expression
"""


def __mergeable_prod(p1: EqNode, p2: EqNode) -> Optional[EqNode]:
    # e.g., Pz(a,x|b,c)Pz(b|c) = Pz(a,x,b|c)
    if not isinstance(p1, ProbTerm) or not isinstance(p2, ProbTerm):
        return None

    if p1.P != p2.P:
        return None

    common_condition = set(p1.condition) & set(p2.condition)
    if set(p1.condition) == (set(p2.condition) | set(p2.measurement)):
        # Pz(a,x|b,c)Pz(b|c) = Pz(a,x,b|c)
        # p1         p2
        return Pr(p1.P, set(p1.measurement) | set(p2.measurement), common_condition)
    elif set(p2.condition) == (set(p1.condition) | set(p1.measurement)):
        # Pz(a,x|b,c)Pz(b|c) = Pz(a,x,b|c)
        # p2         p1
        return Pr(p1.P, set(p2.measurement) | set(p1.measurement), common_condition)

    return None


def __mergeable_div(top: EqNode, bot: EqNode) -> Optional[EqNode]:
    if not isinstance(top, ProbTerm) or not isinstance(bot, ProbTerm):
        return None

    output = __sub_mergeable_div(top, bot)
    if output:
        return output

    inv_output = __sub_mergeable_div(bot, top)
    if inv_output:
        return frac_node(ONE1, inv_output)

    return None


def __sub_mergeable_div(top: ProbTerm, bot: ProbTerm) -> Optional[EqNode]:
    # e.g., P(a,b | c,d) / P(a | c,d) = P(b | a,c,d)
    if top.condition == bot.condition and set(top.measurement) > set(bot.measurement) and top.P == bot.P:
        difference = set(top.measurement) - set(bot.measurement)
        common = set(top.measurement) & set(bot.measurement)
        return Pr(top.P, difference, common | set(top.condition))

    return None


# noinspection DuplicatedCode
def __split_tops_and_bottoms(node: EqNode, outer_change: MutableBoolean) -> Tuple[List[EqNode], List[EqNode]]:
    if isinstance(node, ProductNode) or isinstance(node, FracNode):
        if isinstance(node, ProductNode):
            tops = list(node.children)
            bottoms = []
        else:
            tops = [node.top]
            bottoms = [node.bottom]
        changed = True
        while changed:
            changed = False
            for i, term in enumerate(list(tops)):
                if isinstance(term, ProductNode):
                    tops.extend(term.children)
                    tops.pop(i)
                    changed = True
                    break
                if isinstance(term, FracNode):
                    tops.append(term.top)
                    bottoms.append(term.bottom)
                    tops.pop(i)
                    changed = True
                    break
            for i, term in enumerate(list(bottoms)):
                if isinstance(term, ProductNode):
                    bottoms.extend(term.children)
                    bottoms.pop(i)
                    changed = True
                    break
                if isinstance(term, FracNode):
                    bottoms.append(term.top)
                    tops.append(term.bottom)
                    bottoms.pop(i)
                    changed = True
                    break

        for t_i, top in enumerate(list(tops)):
            for b_i, bottom in enumerate(list(bottoms)):
                if top == bottom:
                    bottoms[b_i] = None
                    tops[t_i] = None
                    outer_change.make_true()
                    break
        tops = notnone(tops)
        bottoms = notnone(bottoms)
        return tops, bottoms
    else:
        return [node], []


def __expand_nested_product(root: EqNode, changed: MutableBoolean) -> EqNode:
    root = root.updated([__expand_nested_product(ch, changed) for ch in root.children])

    if isinstance(root, ProductNode):
        if any(isinstance(ch, ProductNode) for ch in root.children):
            changed.make_true()
            expanded = []
            for ch in root.children:
                if isinstance(ch, ProductNode):
                    expanded.extend(ch.children)
                else:
                    expanded.append(ch)
            return Π(expanded)

    return root


def __move_const_out_of_frac(root: EqNode, changed: MutableBoolean) -> EqNode:
    # T/ k = 1/k * T
    # k / T = k * 1/T
    root = root.updated([__move_const_out_of_frac(ch, changed) for ch in root.children])

    if isinstance(root, FracNode):
        if isinstance(root.top, ConstantNode) and root.top != ONE1:
            changed.make_true()
            return Π([root.top, frac_node(ONE1, root.bottom)])

        if isinstance(root.bottom, ConstantNode) and root.bottom != ONE1:
            changed.make_true()
            return Π([root.bottom.inversed(), root.top])

    return root


def __contract_constant(root: EqNode, changed: MutableBoolean) -> EqNode:
    # C1 * C2 = C3
    # 1 * T = T
    # T / 1 = T
    # 1 / 1 = 1
    root = root.updated([__contract_constant(ch, changed) for ch in root.children])

    if isinstance(root, ProductNode):
        # consts, non_consts = split_by(root.children, lambda ch: isinstance(ch, ConstantNode))
        consts, non_consts = class_split_by(root.children, ConstantNode)
        if len(consts) >= 2:
            merged = ConstantNode(mults(const_node.val for const_node in consts))
            changed.make_true()
            return Π([merged, *non_consts])

    return root


def __canceling_out(root: EqNode, changed: MutableBoolean) -> EqNode:
    root = root.updated([__canceling_out(ch, changed) for ch in root.children])

    tops, bots = __split_tops_and_bottoms(root, changed)

    return frac_node(Π(tops), Π(bots))


# noinspection PyTypeChecker
def __contract_terms(root: EqNode, changed: MutableBoolean) -> EqNode:
    # multiplication
    # e.g., # P(a,b|c,d)P(c|d) = P(a,b,c|d)
    # division
    # e.g., P( a, b | c, d) / P(a | c, d) = P(b | a , c, d)
    # e.g., P(a | c, d)  / P( a, b | c, d)= 1 / P(b | a , c, d)

    root = root.updated([__contract_terms(ch, changed) for ch in root.children])

    tops, bots = __split_tops_and_bottoms(root, changed)
    for ll in [tops, bots]:
        for i, t_i in enumerate(list(ll)):
            for j, t_j in enumerate(ll[i + 1:], i + 1):
                new_t = __mergeable_prod(t_i, t_j)
                if new_t:
                    ll[i] = None
                    ll[j] = new_t
                    changed.make_true()
                    break

    tops = notnone(tops)
    bots = notnone(bots)

    for i, t_i in enumerate(list(tops)):
        for j, b_j in enumerate(list(bots)):
            new_t = __mergeable_div(t_i, b_j)
            if new_t:
                if isinstance(new_t, FracNode) and new_t.top == ONE1:
                    tops[i] = None
                    bots[j] = new_t.inversed()
                else:
                    tops[i] = new_t
                    bots[j] = None
                changed.make_true()
                break

    # TODO a term (either normal or inversed) outside a sum can be merged with other term inside the sum if "sum over variables" disjoint to the term.
    tops = notnone(tops)
    bots = notnone(bots)

    return frac_node(Π(tops), Π(bots))


# noinspection PyUnresolvedReferences
def __sum_decompose(root: EqNode, changed: MutableBoolean) -> EqNode:
    # change a single sum to nested sums
    # irrelevant & relevant
    # split sum into multiple sums (which order?)

    root = root.updated([__sum_decompose(ch, changed) for ch in root.children])

    if isinstance(root, SumNode):
        if isinstance(root.child, ProductNode) or isinstance(root.child, FracNode):
            tops, bots = __split_tops_and_bottoms(root.child, changed)
            top_irres, top_reles = split_by(tops, lambda ch: set(ch.defined_over).isdisjoint(root.sum_over_vars))
            bot_irres, bot_reles = split_by(bots, lambda ch: set(ch.defined_over).isdisjoint(root.sum_over_vars))

            if top_irres or bot_irres:
                changed.make_true()

                return Π([frac_node(Π(top_irres),
                                    Π(bot_irres)),
                          Σ(root.sum_over_vars,
                            frac_node(Π(top_reles),
                                      Π(bot_reles)))])

            tops, bots = __split_tops_and_bottoms(root.child, changed)
            for sum_var in root.sum_over_vars:
                top_reles, top_irres = split_by(tops, lambda t: sum_var in set(t.defined_over))
                bot_reles, bot_irres = split_by(bots, lambda t: sum_var in set(t.defined_over))

                if top_irres or bot_irres:
                    changed.make_true()

                    return Σ(set(root.sum_over_vars) - {sum_var},
                             Π([frac_node(Π(top_irres),
                                          Π(bot_irres)),
                                Σ({sum_var},
                                  frac_node(Π(top_reles),
                                            Π(bot_reles)))]))

    return root


def __drop_condition(root: EqNode, changed: MutableBoolean, G: CausalDiagram = None) -> EqNode:
    if G is None:
        return root

    root = root.updated([__drop_condition(ch, changed, G) for ch in root.children])

    # e.g. Px(y|z,w) = Px(y|z)
    if isinstance(root, ProbTerm):
        if G is not None:
            del_obs = DoCalculus(G).deletable_observation(root.as_probspec())
            if del_obs:
                changed.make_true()
                return ProbTerm(root.P, set(root.measurement), set(root.condition) - del_obs)

    return root


def rewriter(root: EqNode, G: CausalDiagram = None) -> EqNode:
    changed = MutableBoolean(True)
    while changed:
        changed.make_false()
        root = __rewriter1(root, changed)

    changed = MutableBoolean(True)
    while changed:
        changed.make_false()
        root = __rewriter2(root, changed, G)

    return root


def __rewriter1(root: EqNode, changed: MutableBoolean) -> EqNode:
    root = __expand_nested_product(root, changed)
    root = __move_const_out_of_frac(root, changed)
    root = __contract_constant(root, changed)
    root = __canceling_out(root, changed)
    root = __sum_decompose(root, changed)

    return root


def __rewriter2(root: EqNode, changed: MutableBoolean, G: CausalDiagram = None) -> EqNode:
    root = __contract_terms(root, changed)
    root = __drop_condition(root, changed, G)
    #
    root = __rewriter1(root, changed)
    return root
