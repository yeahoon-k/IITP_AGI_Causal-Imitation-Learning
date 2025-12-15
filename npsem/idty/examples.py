import functools
from dataclasses import dataclass, field
from typing import AbstractSet, List, Tuple, FrozenSet, Collection, Sequence

from npsem.idty.gid import is_gID, is_gIDC
from npsem.idty.gtr import is_gTRC
from npsem.idty.transportability import is_TR
from npsem.idty.zid import is_zID
from npsem.model import CausalDiagram, qcd, CD

VSet = FrozenSet[str]
ASet = AbstractSet[str]


@dataclass
class TRExample:
    is_tr: bool
    G: CausalDiagram
    Ys: ASet
    Xs: ASet
    Ss: AbstractSet = field(default_factory=set)
    comment: str = ''

    def check(self):
        assert bool(is_TR(self.G, self.Ys, self.Xs, self.Ss)) == self.is_tr


@dataclass
class ZIDExample:
    is_zid: bool
    G: CausalDiagram
    Ys: ASet
    Xs: ASet
    Zs: ASet = field(default_factory=set)
    comment: str = ''

    def check(self):
        assert is_zID(self.G, self.Ys, self.Xs, self.Zs) == self.is_zid

    def __iter__(self):
        return iter([self.G, self.Ys, self.Xs, self.Zs])


@dataclass
class GIDExample:
    is_gid: bool
    G: CausalDiagram
    Ys: ASet
    Xs: ASet
    Zss: Collection[ASet]
    comment: str = ''

    def check(self):
        assert is_gID(self.G, self.Ys, self.Xs, self.Zss) == self.is_gid

    def __iter__(self):
        return iter([self.G, self.Ys, self.Xs, self.Zss])


@dataclass
class GIDCExample:
    is_gidc: bool
    G: CausalDiagram
    Ys: ASet
    Xs: ASet
    Ws: ASet
    Zss: Collection[ASet]
    comment: str = ''

    def check(self):
        assert is_gIDC(self.G, self.Ys, self.Xs, self.Ws, self.Zss) == self.is_gidc

    def __iter__(self):
        return iter([self.G, self.Ys, self.Xs, self.Ws, self.Zss])


@dataclass
class GTRCExample:
    is_gtrc: bool
    G: CausalDiagram
    Ys: ASet
    Xs: ASet
    Ws: ASet
    Zss: Sequence[Collection[ASet]]
    Sss: Sequence[ASet]
    comment: str = ''

    def check(self):
        assert is_gTRC(self.G, self.Ys, self.Xs, self.Ws, self.Zss, self.Sss) == self.is_gtrc

    def __iter__(self):
        return iter([self.G, self.Ys, self.Xs, self.Zss])


def tr_examples() -> List[TRExample]:
    exs = list()
    exs.append(TRExample(True, qcd(['ZXY', 'ZY'], ['ZXY']), {'Y'}, {'X'}, {'Z'}, 'TR-Fig.1(a)'))
    exs.append(TRExample(True, qcd(['UXWZY', 'UTW', 'VX', 'VY'], ['TUXW', 'XY']), {'Y'}, {'X'}, {'U', 'Z'}, 'TR-Fig.1(b)'))
    exs.append(TRExample(True, qcd(['XZWVY'], ['ZXYV']), {'Y'}, {'X'}, {'V'}, 'TR-Fig.1(c)'))
    exs.append(TRExample(False, qcd(['XY'], ['XY']), {'Y'}, {'X'}, {'Y'}, 'TR-Fig.2(a)'))
    exs.append(TRExample(False, qcd(['XZY'], ['XZ']), {'Y'}, {'X'}, {'Z'}, 'TR-Fig.2(b)'))
    exs.append(TRExample(False, qcd(['XABCY'], ['BAXYC']), {'Y'}, {'X'}, {'C'}, 'TR-Fig.3'))
    exs.append(TRExample(False, qcd(['XZY'], ['XYZ']), {'Y'}, {'X'}, {'Z'}, 'TR-Fig.5a'))
    exs.append(TRExample(False, qcd(['ZY', 'XY'], ['XYZ']), {'Y'}, {'X'}, {'Z'}, 'TR-Fig.5b'))
    return exs


def zID_examples_BP2012UAI(is_zid=True) -> List[ZIDExample]:
    """ zID examples as shown in
        Causal Inference by Surrogate Experiments
        E. Bareinboim, J. Pearl.
        UAI-12. In Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence, 2012.
    """
    if is_zid:
        zid_a = ZIDExample(True, qcd(['WZXY', 'WY'], ['XZYW']), {'Y'}, {'X'}, {'Z'}, comment="UAI zID BP Figure 3(a)")
        zid_b = ZIDExample(True, qcd(['WZXY', 'WY'], ['WXZY']), {'Y'}, {'X'}, {'Z'}, comment="UAI zID BP Figure 3(b)")
        zid_c = ZIDExample(True, qcd(['ZXWY'], ['WZXYZ']), {'Y'}, {'X'}, {'Z'}, comment="UAI zID BP Figure 3(c)")
        zid_d = ZIDExample(True, qcd(['ZXAY', 'BX', 'BY'], ['XBZXYZA']), {'Y'}, {'X'}, {'Z'},
                           comment="UAI zID BP Figure 3(d)")
        return [zid_a, zid_b, zid_c, zid_d]
    else:
        zid_e = ZIDExample(False, qcd(['WZXY', 'WY'], ['ZXWYZ']), {'Y'}, {'X'}, {'Z'}, comment="UAI zID BP Figure 3(e)")
        zid_f = ZIDExample(False, qcd(['ZWY', 'ZXY'], ['WXZY']), {'Y'}, {'X'}, {'Z'}, comment="UAI zID BP Figure 3(f)")
        zid_g = ZIDExample(False, qcd(['ZXWY', 'ZY'], ['WZXYZ']), {'Y'}, {'X'}, {'Z'}, comment="UAI zID BP Figure 3(g)")
        zid_h = ZIDExample(False, qcd(['XWZY'], ['ZXYW']), {'Y'}, {'X'}, {'Z'}, comment="UAI zID BP Figure 3(h)")
        return [zid_e, zid_f, zid_g, zid_h]


def zID_examples(is_zid=True) -> Tuple[ZIDExample]:
    out = list(zID_examples_BP2012UAI(is_zid))
    out.extend(ID_examples(is_id=is_zid))
    if is_zid:
        out.append(ZIDExample(True, qcd(['WZXY'], ['ZYWX']), {'Y'}, {'X'}, {'Z'}, 'Lee Thm 3 Counterexample 1'))
        out.append(ZIDExample(True, qcd(['WXZY'], ['ZWYX']), {'Y'}, {'X'}, {'Z'}, 'Lee Thm 3 Counterexample 2'))
        out.append(ZIDExample(True, qcd(['ZWXY', 'ZY'], ['YWZX']), {'Y'}, {'X'}, {'Z'}))
    else:
        pass
    return tuple(out)


def ID_examples(is_id=True) -> Tuple[ZIDExample]:
    out = list()
    if not is_id:
        out.append(ZIDExample(False, qcd(['XY'], ['XY']), {'Y'}, {'X'}, comment="Figure 2a AAAI'06 Shpitser, Pearl"))  #
        out.append(ZIDExample(False, qcd(['XZY'], ['XZ']), {'Y'}, {'X'}, comment="Figure 2b AAAI'06 Shpitser, Pearl"))  #
        out.append(ZIDExample(False, qcd(['XZY', 'XY'], ['XZ']), {'Y'}, {'X'}, comment="Figure 2c AAAI'06 Shpitser, Pearl"))  #
        out.append(ZIDExample(False, qcd(['XY', 'ZY'], ['XZY']), {'Y'}, {'X'}, comment="Figure 2d AAAI'06 Shpitser, Pearl"))  #
        out.append(ZIDExample(False, qcd(['ZXY'], ['XZY']), {'Y'}, {'X'}, comment="Figure 2e AAAI'06 Shpitser, Pearl"))  #
        out.append(ZIDExample(False, qcd(['XZY'], ['ZYX']), {'Y'}, {'X'}, comment="Figure 2f AAAI'06 Shpitser, Pearl"))  #
        out.append(ZIDExample(False, qcd(['XWY', 'QY'], ['XQW']), {'Y'}, {'X'}, comment="Figure 2g AAAI'06 Shpitser, Pearl"))  #
        out.append(ZIDExample(False, qcd(['ZXWY'], ['ZXYZW']), {'Y'}, {'X'}, comment="Figure 2h AAAI'06 Shpitser, Pearl"))  #
        out.append(ZIDExample(False, qcd(['WZXY'], ['WX', 'ZY', 'WY']), {'Y'}, {'X'}, comment=""))
        out.append(ZIDExample(False, qcd(['WZXY'], ['WX', 'ZY', 'WY']), {'Y'}, {'Z'}, comment=""))
    else:
        out.append(ZIDExample(True, qcd(['XAY', 'BX', 'BA', 'BCY'], ['CXYBX']), {'Y'}, {'X'}, comment="Figure 1 AAAI'02 Tian, Pearl"))  #
        out.append(ZIDExample(True, qcd(['XABY', 'XY', 'AY'], ['XB', 'AY']), {'Y'}, {'X'}, comment="Figure 2 AAAI'02 Tian, Pearl"))  #

    return tuple(out)


@functools.lru_cache(maxsize=1)
def GID_examples(is_gid=True) -> Tuple[GIDExample]:
    out = []
    if is_gid:
        # CROSS CRAB
        out.append(GIDExample(True, qcd(['AC', 'BD'], ['ADCB'], u_names=['U_AD', 'U_DC', 'U_BC']), set('CD'), set('AB'), [{'A', 'B'}]))
        out.append(GIDExample(True, qcd(['AC', 'BD'], ['ADCB'], u_names=['U_AD', 'U_DC', 'U_BC']), set('CD'), set('AB'), [{'A', 'B'}, {'A'}, {'B'}, set()]))
        pass
    else:
        out.append(GIDExample(False, qcd(['YW', 'ZVT', 'YUT'], ['UXT', 'TWY', 'VX', 'XZ']), frozenset({'T', 'W', 'X'}), frozenset({'U'}), [set()]))

        # CROSS CRAB
        out.append(GIDExample(False, qcd(['AC', 'BD'], ['ADCB'], u_names=['U_AD', 'U_DC', 'U_BC']), set('CD'), set('AB'), [set()]))
        out.append(GIDExample(False, qcd(['AC', 'BD'], ['ADCB'], u_names=['U_AD', 'U_DC', 'U_BC']), set('CD'), set('AB'), [{'B'}, {'A'}]))
        out.append(GIDExample(False, qcd(['AC', 'BD'], ['ADCB'], u_names=['U_AD', 'U_DC', 'U_BC']), set('CD'), set('AB'), [{'B'}, {'A'}, set()]))
        out.append(GIDExample(False, qcd(['ACE', 'BDE'], ['ADCB'], u_names=['U_AD', 'U_DC', 'U_BC']), set('E'), set('AB'), [{'B'}, {'A'}]))
        out.append(GIDExample(False, qcd(['ACE', 'BDF'], ['ADCB'], u_names=['U_AD', 'U_DC', 'U_BC']), set('EF'), set('AB'), [{'B'}, {'A'}]))

        out.append(GIDExample(False, gid_figure3a(), {'R'}, {'X1', 'X2'}, [{'X1'}, {'X2'}]))
        out.append(GIDExample(False, gid_figure3b(), {'R1', 'R2'}, {'X1', 'X2'}, [{'X1'}, {'X2'}]))
        out.append(GIDExample(False, gid_figure3c(), {'R'}, {'X1', 'X2'}, [{'X1'}, {'X2'}]))

        out.append(GIDExample(False, gid_figure4a(), set('HIJK'), {'B'}, [{'G'}, {'E'}]))
        out.append(GIDExample(False, gid_figure4a(), set('HIJK'), {'E', 'G', 'F'}, [set('F'), set('A')]))
        out.append(GIDExample(False, gid_figure4a(), set('HIJK'), {'A', 'D'}, [set('FGD'), set('EA')]))

        out.append(GIDExample(False, gid_figure4a(), set('HIJK'), {'A', 'F', 'G'}, [set('FD'), set('A')]))
        pass

    return tuple(out)


def gid_figure3a() -> CausalDiagram:
    return qcd([['X1', 'R'], ['X2', 'R']], [['X1', 'R', 'X2']])


def gid_figure3b() -> CausalDiagram:
    return qcd([['X1', 'R1'], ['X2', 'R2']], [['X1', 'R2', 'R1', 'X2']])


def gid_figure3c() -> CausalDiagram:
    return qcd([['W', 'X2', 'R'], ['W', 'X1', 'R']], [['X2', 'W', 'X1'], 'RW'])


def gid_figure3d() -> CausalDiagram:
    # R1,R2,R3 = ABC
    # Ti = i
    G = qcd(['5321A', '46B'], ['14CBA', '6253B'])
    return G


def gid_figure4a() -> CausalDiagram:
    # R1, R2, R3, R4 = H I J K
    return qcd(['EAH', 'FBH', 'CJ', 'GDK'], ['KIHECBA', 'IJ', 'JFD', 'CGK'])


def gid_figure4b() -> CausalDiagram:
    # R1, R2, R3, R4 = H I J K
    return qcd(['EAH', 'BH', 'CJ'], ['KIHECBA', 'IJ'])


def gid_figure4c() -> CausalDiagram:
    # R1, R2, R3, R4 = H I J K
    return qcd(['FBH', 'DK'], ['HIJFD', 'IK'])


def gid_figure4d() -> CausalDiagram:
    # R1, R2, R3, R4 = H I J K
    return qcd(['BH', 'CJ', 'GDK'], ['HIKGCB', 'IJ'])


def gid_figure4s():
    return [gid_figure4a(),
            gid_figure4b(),
            gid_figure4c(),
            gid_figure4d()]


def gidc_examples():
    problems = []

    G = qcd('Y<->W<->X->W')
    Ws = {'W'}
    Ys = {'Y'}
    Xs = {'X'}
    Zss = [frozenset()]
    problems.append(GIDCExample(True, G, set(), Xs, Ws, []))
    problems.append(GIDCExample(False, G, Ys, Xs, Ws, Zss))

    G = qcd([['X1', 'X2', 'W1'], ['V', 'W2', 'X2']], [['X2', 'V', 'X1', 'W2'], ['Y', 'V', 'W1']])
    Ws = {'W1', 'W2'}
    Ys = {'Y'}
    Xs = {'X1', 'X2'}
    Zss = [frozenset()]
    problems.append(GIDCExample(False, G, Ys, Xs, Ws, Zss))

    G = qcd(['ZXT', 'ZY', 'UT'], ['TVYWXU', 'XZ'])
    # G = qcd('Y<-Z->X->T<-U, T<->V<->Y<->W<->X<->U, X<->Z')
    Ys = {'V'}
    Xs = set()
    Ws = {'Y', 'T'}
    Zss = [{'Z'}]
    problems.append(GIDCExample(False, G, Ys, Xs, Ws, Zss))

    G = qcd(['YWU', 'YX', 'YT', 'VU', 'ZU'], ['VTXZUW', 'UV'])
    Ys = {'W', 'Z'}
    Xs = {'U'}
    Ws = {'T', 'V', 'X'}
    Zss = [{'Y'}]
    problems.append(GIDCExample(False, G, Ys, Xs, Ws, Zss))

    G = qcd([['X1', 'X2', 'W1'], ['V', 'W2', 'X2']], [['X2', 'V', 'X1', 'W2'], ['Y', 'V', 'W1']])
    Ws = {'W1', 'W2'}
    Ys = {'Y'}
    Xs = {'X1', 'X2'}
    Zss = [set()]
    problems.append(GIDCExample(False, G, Ys, Xs, Ws, Zss))

    G = qcd(['ADE', 'BCF', 'XVD', 'VC'], ['VBFEAVXD', 'XC', 'FY'])
    Ys = {'Y'}
    Xs = {'X'}
    Ws = {'E', 'F'}
    Zss = [{'D'}, {'C'}]
    problems.append(GIDCExample(False, G, Ys, Xs, Ws, Zss))

    return problems


@functools.lru_cache()
def gid_examples():
    fig1a = GIDExample(True, CD('X1->W->Y<-X2, W<->X1<->X2<->Y'), {'Y'}, {'X1', 'X2'}, [{'X1'}, {'X2'}])
    fig1b = GIDExample(True, CD('X1->W->Y<-X2, W<->X1<->X2<->Y, W<->X2'), {'Y'}, {'X1', 'X2'}, [{'X1'}, {'X2'}])
    fig1c = GIDExample(False, CD('X1->W->Y<-X2, W<->X1<->X2<->Y, W<->Y'), {'Y'}, {'X1', 'X2'}, [{'X1'}, {'X2'}])
    fig1d = GIDExample(False, CD('X1->W->Y<-X2, W<->X1<->X2<->Y, X1<->Y'), {'Y'}, {'X1', 'X2'}, [{'X1'}, {'X2'}])

    # P(y1|do(x1))
    fig2a_1 = GIDExample(True, CD('X1->Y1, X2->Y2, X1<->Y1<->X2<->Y2<->X1<->X2, Y1<->Y2'), {'Y1'}, {'X1'}, [{'X1', 'X2'}])
    fig2b_1 = GIDExample(True, CD('X1->Y1, X2->Y2, X1<->Y1<->X2<->Y2<->X1<->X2, Y1<->Y2, Y1->Y2'), {'Y1'}, {'X1'}, [{'X1', 'X2'}])
    fig2c_1 = GIDExample(True, CD('X1->Y1, X2->Y2, X1<->Y1<->X2<->Y2<->X1<->X2, Y1<->Y2, X1->Y2'), {'Y1'}, {'X1'}, [{'X1', 'X2'}])
    # P(y2|do(x2))
    fig2a_2 = GIDExample(True, CD('X1->Y1, X2->Y2, X1<->Y1<->X2<->Y2<->X1<->X2, Y1<->Y2'), {'Y2'}, {'X2'}, [{'X1', 'X2'}])
    fig2b_2 = GIDExample(False, CD('X1->Y1, X2->Y2, X1<->Y1<->X2<->Y2<->X1<->X2, Y1<->Y2, Y1->Y2'), {'Y2'}, {'X2'}, [{'X1', 'X2'}])
    fig2c_2 = GIDExample(False, CD('X1->Y1, X2->Y2, X1<->Y1<->X2<->Y2<->X1<->X2, Y1<->Y2, X1->Y2'), {'Y2'}, {'X2'}, [{'X1', 'X2'}])

    fig3a = GIDExample(False, CD('X1->R<-X2, X1<->R<->X2'), {'R'}, {'X1', 'X2'}, [{'X1'}, {'X2'}])
    fig3b = GIDExample(False, CD('X1->R1, X2->R2, X1<->R2<->R1<->X2'), {'R1', 'R2'}, {'X1', 'X2'}, [{'X1'}, {'X2'}])
    fig3c = GIDExample(False, CD('W->X1->R, W->X2->R, R<->W<->X2, X1<->W'), {'R'}, {'X1', 'X2'}, [{'X1'}, {'X2'}])

    fig3d = GIDExample(False, CD('T5->T3->T2->T1->R1, T4->T6->R2, R1<->R2<->R3<->T4<->T1, R2<->T3<->T5<->T2<->T6'), {'R1', 'R2', 'R3'}, {'T1', 'T6'}, [set()])

    return {
        'fig1a': fig1a,
        'fig1b': fig1b,
        'fig1c': fig1c,
        'fig1d': fig1d,

        'fig2a_1': fig2a_1,
        'fig2a_2': fig2a_2,
        'fig2b_1': fig2b_1,
        'fig2b_2': fig2b_2,
        'fig2c_1': fig2c_1,
        'fig2c_2': fig2c_2,

        'fig3a': fig3a,
        'fig3b': fig3b,
        'fig3c': fig3c,

        'fig3d': fig3d,
    }
