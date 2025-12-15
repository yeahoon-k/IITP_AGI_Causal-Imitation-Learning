import itertools
from typing import Union, Iterable

import numpy as np

from npsem.idty.gid import gID, Thicket
from npsem.idty.gidpo.jigsaw import CausalJigsaw
from npsem.idty.idty_utils import NotIdentifiableError
from npsem.model import qcd, SCM, CD, scm
from npsem.model_utils import default_P_U
from npsem.model_utils import print_prob_result
from npsem.parsers import parse_probspec, ensure_probspec
from npsem.stat_utils import ProbSpec
from npsem.utils import subsets


def _wrap_probspecs(xs):
    if isinstance(xs, str):
        return [parse_probspec(xs)]
    elif isinstance(xs, ProbSpec):
        return [xs]
    else:
        return list(map(ensure_probspec, xs))


# TODO change name
def compare_distributions(M1: SCM,
                          M2: SCM,
                          test_probs: Union[str, ProbSpec, Iterable[Union[str, ProbSpec]]],
                          check_equality=True,
                          *,
                          check_positivity=False,
                          verbose=False):
    """
    check_positivity only if check_equality
    """
    test_probs = _wrap_probspecs(test_probs)
    for prob in [test_probs] if isinstance(test_probs, ProbSpec) else test_probs:
        xss = list(itertools.product(*[M1.D[X] for X in prob.X]))
        Xx, P1, P2 = None, None, None
        for xs in xss:
            Xx = dict(zip(prob.X, xs))
            P1 = M1.query(prob.Y, intervention=Xx)
            P2 = M2.query(prob.Y, intervention=Xx)

            if verbose:
                print(prob, Xx)
                print_prob_result([P1, P2], prob.Y, M1.D)

            if check_equality:
                if not all(np.allclose(P1[k], P2[k]) for k in P1.keys() | P2.keys()):
                    return False
                if check_positivity:
                    if not all(P1[k] > 0 and P2[k] > 0 for k in M1.values_of(prob.Y)):
                        return False
            else:
                if not all(np.allclose(P1[k], P2[k]) for k in P1.keys() | P2.keys()):
                    break
        else:
            if not check_equality:
                if verbose:
                    print(prob, Xx)
                    print_prob_result([P1, P2], prob.Y, M1.D)
                return False

    return True


def test_fig3():
    G = CD('X3->X2->D, C->X2->A->Y1->Y2, A<-B->E, B->X1->Y2, X1<-Y3->X4')
    cj = CausalJigsaw('P(Y1,Y2,Y3|do(X1,X2,X3,X4))', G, data=[])
    assert cj.Ys_star == cj.Ys
    assert cj.Xs_star == {'X1', 'X2'}
    assert cj.Xs_plus == {'X1', 'X2', 'X3', 'C'}
    assert cj.Ys_plus == {'Y1', 'Y2', 'Y3', 'A', 'B'}
    assert cj.Vs_plus == cj.Xs_plus | cj.Ys_plus
    assert cj.Vs_star == cj.Xs_star | cj.Ys_star


def catch_thicket(_) -> Thicket:
    try:
        _()
        assert False
    except NotIdentifiableError as e:
        assert isinstance(e.evidence, Thicket)
        return e.evidence


def test_fig4():
    # 4a
    G = CD('X->Y<->X')
    thicket = catch_thicket(lambda: gID(G, {'Y'}, {'X'}, [set()]))
    # print(thicket)
    assert len(thicket.hedges) == 1
    assert thicket.Rs == {'Y'}

    # 4b
    G = CD('W->X->Z->Y, Z<->W<->Y<->X')
    thicket = catch_thicket(lambda: gID(G, {'Y'}, {'X'}, [set()]))
    # print(thicket)
    assert len(thicket.hedges) == 1
    assert thicket.Rs == {'Y'}

    # 4c
    G = CD('X->Y1, W->Y2, Y1<->W<->X<->Y2')
    thicket = catch_thicket(lambda: gID(G, {'Y1', 'Y2'}, {'X'}, [set()]))
    # print(thicket)
    assert len(thicket.hedges) == 1
    assert thicket.Rs == {'Y1', 'W'}


def test_fig5():
    G = CD('Z->X->Y<-W, Y<->W<->Z<->X')
    cj = CausalJigsaw(parse_probspec('P(Y|do(X))'), G, ['P(X,Y|do(Z))'])
    assert cj.is_identifiable()
    assert str(cj.identify()) == 'P_{z}(y | x)'


def spec2dict(ps):
    return {V: V.lower() for V in ps.Xs | ps.Ys | ps.Zs}


def test_fig6():
    verbosity = False  # change if needed to see distributions

    G = qcd(['XY', 'AY', 'BY'], ['XAB'], u_names=['U1', 'U2'])
    funcstr1 = """
        X = U1
        A = U1 ⊕ U2
        B = U2
        Y = X ⊕ A ⊕ B ⊕ U_Y
    """
    funcstr2 = """
        X = U1
        A = U1 ⊕ U2
        B = U2
        Y = U_Y
    """

    def p_u(uu):
        # internally, all other unobserved variables are fair coins
        return 0.9 if uu['U_Y'] == 0 else 0.1

    M1 = SCM(G, F=funcstr1, P_U=p_u, more_U={'U_Y'})
    M2 = SCM(G, F=funcstr2, P_U=p_u, more_U={'U_Y'})

    assert compare_distributions(M1, M2, ['P(X,Y,A)', 'P(X,Y,B)'], verbose=verbosity)
    assert compare_distributions(M1, M2, 'P(Y|do(X))', check_equality=False, verbose=verbosity)


def test_figure_1_and_7():
    G = qcd(['AXBCY', 'AB'], ['XY'])

    cj = CausalJigsaw('P(Y|do(X))', G, data=['P(A,C|do(X))', 'P(B,Y,C)'])
    cj.chart.print()
    print(cj.is_identifiable())
    print(cj.identify().str2({'Y': 'y', 'X': 'x'}))

    print('----------')
    for ff in map(parse_probspec, ["P(A)", "P(Y|do(C))", "P(C|do(B))", "P(B|do(A,X))"]):
        print(ff)
        cj = CausalJigsaw(ff, G, data=['P(A,C|do(X))', 'P(B,Y,C)'])
        print(cj.is_identifiable())
        if cj.is_identifiable():
            print(cj.identify().str2(spec2dict(ff)))
        print()

    print('----------')

    for ff in map(parse_probspec, ["P(A)", "P(Y|do(C))", "P(C|do(A,X))"]):
        cj = CausalJigsaw(ff, G, data=['P(A,C|do(X))', 'P(B,Y,C)'])
        print(cj.is_identifiable())
        if cj.is_identifiable():
            print(cj.identify().str2(spec2dict(ff)))
        print()


def ffstr(ff):
    if ff is None:
        return None
    if ff[0]:
        return f'P_{{{("".join(sorted(ff[0]))).lower()}}}({("".join(sorted(ff[1]))).lower()})'
    else:
        return f'P({("".join(sorted(ff[1]))).lower()})'


def test_fig8():
    G = qcd(['AXBCY', 'AB'], ['XY'])
    cj = CausalJigsaw('P(Y|do(X))', G, data=['P(A,C|do(X))', 'P(B,Y,C)'])

    examine_embedding_relationships(cj)
    examine_mvefs(cj)


def examine_mvefs(cj):
    print('enumerate MVEFs')
    for Zs, Vs in zip(cj.data.Zss, cj.data.Vss):  # given data
        print(f'distribution: {ProbSpec(Zs, Vs)}')
        for factor in cj.gc_factors():  # \mathbb{F}
            mvef_info = cj.MVEF(factor[0], factor[1], Zs, Vs)
            if mvef_info:
                vars_marginalized_out, mvef = mvef_info
                if vars_marginalized_out:
                    print(f'  MVEF {ffstr(mvef)} for a factor {ffstr(factor)} by projecting out {set(vars_marginalized_out)}.')
                else:
                    print(f'  MVEF {ffstr(mvef)} = a factor {ffstr(factor)}.')


def examine_embedding_relationships(cj):
    # embed_relationships = defaultdict(set) # Hasse diagram
    import networkx as nx
    nxG = nx.DiGraph()
    for out_vars in subsets(cj.Vs_plus - cj.Vs_star):
        # one less outvar
        for one_var in out_vars:
            one_less_out_vars = out_vars - {one_var}
            embeds_rels = cj.embedding_factor_map(one_less_out_vars, out_vars)
            for k, v in embeds_rels.items():
                if v != k:
                    nxG.add_edge(k, v)
    nxG = nx.transitive_reduction(nxG)
    for v in nxG:
        print(ffstr(v).rjust(10), '-->', [ffstr(_) for _ in list(nxG.successors(v))])


def test_fig9():
    G = CD('A->D->Y1<-E->Y2<-B, X->D, A<->B, Y1<->E, X<->Y2')
    cj = CausalJigsaw('P(Y1,Y2|do(X))', G, data=['P(A,X,Y1,Y2|do(B))'])

    examine_embedding_relationships(cj)
    examine_mvefs(cj)


def test_fig10():
    pass


def test_fig11():
    # Figure 11(a)
    G = CD('P->W->C1, W->C2, N1<->W<->N2')
    H = G.marginalize_out('W')
    print(H)
    # Figure 11(b)
    assert H == CD('C1<-P->C2<->C1<->N1<->C2<->N2<->C1')


def test_section_B_example1():
    epsilon = 0.4
    p_u1 = default_P_U({'U': 0.5, 'U_Z': epsilon})
    p_u2 = default_P_U({'U': 0.5, 'U_Z': epsilon, 'U_Y': epsilon})
    M1 = scm("""
            X = U
            Z = X ^ U_Z
            Y = Z ^ U
            """, p_u1)
    M2 = scm("""
            X = U
            Z = X ^ U_Z
            Y = U_Y
            """, p_u2)

    assert compare_distributions(M1, M2, ['P(X,Y)'])
    assert compare_distributions(M1, M2, 'P(Y|do(X))', check_equality=False)
