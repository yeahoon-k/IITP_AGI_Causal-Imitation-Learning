from typing import FrozenSet, AbstractSet

from npsem.idty.exp_rewrite import rewriter
from npsem.idty.idty_utils import NotIdentifiableError, is_c_forest, is_super_c_forest
from npsem.idty.prob_eq import ProbDistr, EqNode, Σ, Pr, Π, ProbTerm, SumNode, assign, ONE1
from npsem.model import CD
from npsem.model_utils import trim_edges_while

VSet = FrozenSet[str]
ASet = AbstractSet[str]


def CC(G: CD):
    return G.c_components


class Hedge:
    """ Hedge is a graphical object which witnesses non-identifiability of a query given an observational distribution in a causal graph
    (c.f. in this class implementation, F' is restricted to sinks.)

    References
    ----------
    Ilya Shpitser, Judea Pearl (2006) Identification of joint interventional distributions in recursive semi-Markovian causal models
    Proceedings of the National Conference on Artificial Intelligence
    """

    def __init__(self, F: CD, F_V: AbstractSet[str]):
        F_ = F[F_V]
        Rs = F.sinks()

        assert Rs
        assert F_.sinks() == Rs, 'both do not share the same root set.'
        assert is_c_forest(F, Rs) and is_c_forest(F_, Rs)

        self.F = F
        self.F_ = F_
        self.Rs = Rs

    def __hash__(self):
        return hash((self.F, self.F_, self.Rs))

    def __eq__(self, other):
        if isinstance(other, Hedge):
            return self.F == other.F and self.F_ == other.F_ and self.Rs == other.Rs

    def __str__(self):
        return f"{str(self.F)} with V(F') is {set(self.F_.V)} over R = {set(self.Rs)}"

    def __repr__(self):
        return f'Hedge({repr(self.F)}, {set(self.F_.V)})'


def __make_hedge(G: CD, Ys: ASet, Xs: ASet, Zs: ASet = frozenset()) -> Hedge:
    assert (G - Xs).is_one_cc
    assert G.sinks() <= Ys
    assert Xs.isdisjoint(Zs), f"unused experiment {Zs & Xs} exists"

    super_Ys = G.V - Xs
    SCF = is_super_c_forest
    F = trim_edges_while(lambda H: SCF(H, super_Ys) and SCF(H[super_Ys], super_Ys), G)
    return Hedge(F, super_Ys)


def zID(G: CD, Ys: ASet, Xs: ASet, Zs: ASet = frozenset(), assigned=None, simplify=True) -> EqNode:
    """
    Elias Bareinboim, Judea Pearl. (2012) Causal Inference by Surrogate Experiments: z-Identifiability
    In Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence

    The original pseudo-code is not detailed enough to be directly translatable to a code.
    Hence, a slight modification is made to be functional and faithful to the authors' original intentions.
    """
    return __zID0(G, ProbDistr(set(), G.V), Ys, Xs, Zs, assigned, simplify)


def __zID0(G: CD, P_I: EqNode, Ys: ASet, Xs: ASet, Zs: ASet = frozenset(), assigned=None, simplify=True) -> EqNode:
    """ z-Identifiability """
    if not Ys:
        return ONE1

    if assigned is None:
        assigned = Ys | Xs

    ret = __zID__(G, P_I, Ys, Xs, Zs, assigned)

    if simplify:
        return rewriter(ret, G)

    return ret


def __zID__(G: CD, P_I: EqNode, Ys: ASet, Xs: ASet, Zs: ASet, assigned: VSet) -> EqNode:
    """ z-Identifiability """
    Vs = G.V  # alias
    ordered = G.causal_order()

    def naive_prevs(qq):
        return set(ordered[:ordered.index(qq)])

    if not Xs:
        assert Ys <= assigned
        formula = Σ(Vs - Ys, Pr(P_I, Vs))
        return formula

    if Vs - G.An(Ys):
        formula = __zID__(G[G.An(Ys)], Σ(Vs - G.An(Ys), P_I), Ys, Xs & G.An(Ys), Zs & G.An(Ys), assigned)  # done
        return formula

    if Ws := (Vs - Xs) - G.do(Xs).An(Ys):
        assert Ws.isdisjoint(assigned)
        formula = assign(__zID__(G, P_I, Ys, Xs | Ws, Zs, assigned | Ws), Ws)  # done
        return formula

    if len(CCs := CC(G - Xs)) > 1:
        terms = [__zID__(G, P_I, S_i, Vs - S_i, Zs, assigned | (Vs - (Xs | Ys))) for S_i in sorted(CCs)]
        formula = Σ(Vs - (Xs | Ys), Π(terms))
        return formula
    else:
        S, = CCs
        if CC(G) != {Vs}:
            if S in CC(G):
                subformula = Π([Pr(P_I, {Q}, naive_prevs(Q), scope=Vs) for Q in sorted(S)])
                formula = Σ(S - Ys, subformula)
                return formula

            for S__ in CC(G):
                if S < S__:
                    new_p = Π([Pr(P_I, {Vi}, naive_prevs(Vi), scope=Vs) for Vi in sorted(S__)])
                    formula = __zID__(G[S__], new_p, Ys, Xs & S__, Zs, assigned)
                    return formula
        else:
            if ZnX := Zs & Xs:
                formula = __zID__(G - ZnX, __activate_experiment(P_I, ZnX), Ys, Xs - ZnX, Zs - ZnX, assigned)  # done
                return formula

            raise NotIdentifiableError('not identifiable', __make_hedge(G, Ys, Xs, Zs))


def __sub_new_activate_experiment(root: EqNode, to_do: ASet) -> EqNode:
    if isinstance(root, ProbDistr):
        assert to_do.isdisjoint(set(root.intervention))
        return ProbDistr(root.intervention | to_do, root.defined_over)

    elif isinstance(root, ProbTerm):
        P_I = __sub_new_activate_experiment(root.P, to_do)
        return Pr(P_I, frozenset(root.measurement) - to_do, frozenset(root.condition) - to_do)

    elif isinstance(root, SumNode):
        # any common variables will be fixed to some values (related to a specified x determined outside the scope)
        # it is equivalent to fixing those variables, hence, no summation over the variables!
        common = set(to_do) & set(root.sum_over_vars)
        return Σ(set(root.sum_over_vars) - common, __sub_new_activate_experiment(root.child, to_do))

    else:
        return root.updated([__sub_new_activate_experiment(ch, to_do) for ch in root.children])


def __activate_experiment(P_I: EqNode, to_do: ASet) -> EqNode:
    # return sub_new_activate_experiment(rewriter(P_I), to_do)
    # should work without rewriter.
    return __sub_new_activate_experiment(P_I, to_do)


def is_ID(G: CD, Ys: ASet, Xs: ASet) -> bool:
    """ Decision version of traditional causal effect identifiability from an observational distribution """
    return is_zID(G, Ys, Xs)


def is_zID(G: CD, Ys: ASet, Xs: ASet, Zs: ASet = frozenset()) -> bool:
    """ Decision version of z-identifiability

    See `zID`
    """
    if not Ys:
        return True
    Vs = G.V  # alias

    if not Xs:
        return True

    if Vs - G.An(Ys):
        return is_zID(G[G.An(Ys)], Ys, Xs & G.An(Ys), Zs)

    if Ws := (Vs - Xs) - G.do(Xs).An(Ys):
        return is_zID(G, Ys, Xs | Ws, Zs)

    if len(CCs := CC(G - Xs)) > 1:
        return all(is_zID(G, S_i, Vs - S_i, Zs) for S_i in CCs)
    else:
        S, = CCs
        if CC(G) != {Vs}:
            if S in CC(G):
                return True
            for S__ in CC(G):
                if S < S__:
                    return is_zID(G[S__], Ys, Xs & S__, Zs)
            raise AssertionError('unreachable code')
        else:
            if coXZ := Zs & Xs:
                return is_zID(G - coXZ, Ys, Xs - coXZ, Zs - coXZ)
            return False
