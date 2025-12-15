from collections import defaultdict
from typing import Tuple, Dict, AbstractSet

from npsem.idty.gid import Thicket
from npsem.model import CD, StructuralCausalModel as SCM
from npsem.model_utils import trim_directed_edges_while, find_a_dpath_graph
from npsem.utils import xors, shuffled, domain_with_bits, bits_at, masked

ASet = AbstractSet[str]


def construct_thicket_models(thicket: Thicket, *, noiseless=False, **_) -> Tuple[SCM, SCM, Dict[str, int], str]:
    Rs = thicket.Rs
    Rstar = shuffled(Rs)[0]
    T = thicket.aggregated()
    Rs_UCs = T[Rs].U
    Rs_Us = {f'U_{R}' for R in Rs} if not noiseless else set()  # TODO to use or not?

    crossing_UCs = T.UCs(Rs) - Rs_UCs
    assert Rs_Us.isdisjoint(T.U)
    assert Rs_UCs.isdisjoint(crossing_UCs)

    if T.V == Rs:
        def Rfunc(R, model):
            assert T.UCs(R) <= Rs_UCs

            def func_(vv):
                return xors([vv[U] for U in T.UCs(R)]) ^ (model == 2 and R == Rstar)

            return func_

        domains = {W: domain_with_bits(1) for W in T.V | T.U}  # Rs_Us?
        M1 = SCM(T, F={R: Rfunc(R, 1) for R in Rs}, D=domains, more_v_to_us=defaultdict(set))
        M2 = SCM(T, F={R: Rfunc(R, 2) for R in Rs}, D=domains, more_v_to_us=defaultdict(set))
        return M1, M2, {R: 1 for R in Rs | Rs_UCs}, Rstar

    # but for Rs and Rs_UCs
    HHs_of = defaultdict(list)  # 0 for right-most
    bit_at = defaultdict(lambda: 0)
    for HH_id, HH in enumerate(thicket.hedges):
        for W in (HH.F.V - Rs) | (HH.F.U - Rs_UCs):  # U or V (except Rs_UCs)
            bit_at[(HH_id, W)] = len(HHs_of[W])
            HHs_of[W].append(HH_id)

    assert crossing_UCs <= HHs_of.keys()
    n_bits = {k: len(l) for k, l in HHs_of.items()}
    n_bits.update({R: 1 for R in Rs})
    n_bits.update({U: 1 for U in Rs_UCs})
    n_bits.update({U: 1 for U in Rs_Us})

    # Correa's method
    to_flip = set()
    for HH_id, HH in enumerate(thicket.hedges):
        F = HH.F
        n_crossings = len(F.UCs(Rs) - Rs_UCs)  # This is the number of variables sending bits for its parities to be checked whether the mechanism above is disturbed.
        n_frontiers = len(F.pa(Rs))  # This is the number of variables where the sent bits are delivered which may keep the bit parity or not.

        phi = n_crossings % 2
        psi = n_frontiers % 2

        # # 2020 Aug, original flipping
        # if False:
        #     if phi == psi == 0:
        #         W = next(iter(F.pa(Rs)))
        #         to_flip.add((HH_id, W))
        #     elif phi == psi == 1:
        #         to_flip |= {(HH_id, W) for W in F.pa(Rs)}
        # else:
        if phi == psi:
            W = next(iter(F.pa(Rs)))
            to_flip.add((HH_id, W))

    # noinspection PyShadowingNames
    def read(vv: Dict[str, int], W: str, HH_id: int):
        Wbits = vv[W]
        return (Wbits >> bit_at[(HH_id, W)]) & 1

    def func(V: str, model_id: int):
        assert model_id in {1, 2}

        # noinspection PyShadowingNames
        def fun(vv: Dict[str, int]):
            if V in Rs:
                R = V  # for clarity...
                # all activated?
                flag = all(all(read(vv, W, HH_id) ^ ((HH_id, W) in to_flip)
                               for HH_id in HHs_of[W])
                           for W in T.pa(R) | (T.UCs(R) - Rs_UCs))

                # ( no_black_out & (xored for UCs among Rs w/ whether to flip) ) whether to flip, again, for positivity
                return (flag & (xors([vv[U] for U in T.UCs(R) & Rs_UCs]) ^ (model_id == 2 and R == Rstar))) ^ (vv[f'U_{R}'] if not noiseless else 0)
            else:
                val = 0
                for HH_id in HHs_of[V]:  # per hedge operation
                    F = thicket.hedges[HH_id].F  # type: CD
                    xored = xors(read(vv, W, HH_id) for W in F.pa(V) | F.UCs(V))
                    val += (xored << bit_at[(HH_id, V)])

                return val

        return fun

    domains = {W: domain_with_bits(n_bits[W]) for W in T.V | T.U | Rs_Us}
    MU = 0.05  # no 0.5!

    def P_U(vv):
        num_ones = sum(vv[U] == 1 for U in Rs_Us if U in vv)
        num_zeros = sum(vv[U] == 0 for U in Rs_Us if U in vv)
        return (MU ** num_ones) * ((1 - MU) ** num_zeros)

    M1 = SCM(T, F={V: func(V, 1) for V in T.V}, P_U=P_U, D=domains, more_U=Rs_Us)
    M2 = SCM(T, F={V: func(V, 2) for V in T.V}, P_U=P_U, D=domains, more_U=Rs_Us)

    return M1, M2, n_bits, Rstar


def extended_thicket(G: CD, thicket: Thicket, Xs: ASet, Ws: ASet) -> Tuple[CD, CD]:
    Rs = thicket.Rs
    E = (G - Xs).without_UCs().underbar(Ws)  # no more down to Ws
    E = E[E.De(Rs - Ws) & E.An(Ws)]  # Rs - Ws needs to be mapped to Ws
    subWs = E.sinks()  # only relevant Ws
    assert subWs <= Ws

    # make it forest, please
    E = trim_directed_edges_while(lambda E_: Rs - Ws <= E_.An(subWs), E)
    # variables b/w unused path after trim will be removed...
    E = E[E.De(Rs - Ws) & E.An(subWs)]

    # attach to the thicket
    H = thicket.aggregated() + E
    # w/ extended, extended part, root set
    # This does not hold, see GTR paper
    # assert E.sinks() | (Rs & Ws) == H.sinks()
    return H, E


def extended_thicket_models(M1: SCM, M2: SCM, thicket: Thicket, TEx: CD, Ex: CD, prev_bits: Dict[str, int], noiseless=False):
    Rs = thicket.Rs
    T = thicket.aggregated()
    assert Rs == T.sinks()
    assert Ex.sinks().isdisjoint(Ex.sources())

    cannot_merge_sink = Ex.sinks() & (TEx.V - TEx.sinks())

    Ms = [None, M1, M2]
    new_Vs = Ex.V - Ex.sources()  # where values are
    new_Vs_Us = set()
    if not noiseless:
        new_Vs_Us = {f'Ex_U_{V}' for V in new_Vs}  # let transferring bits be noisy.

    assert Rs <= prev_bits.keys()
    to_be_copied = Rs - TEx.sinks()  # not necessarily the source of Ex!!
    assert Ex.sources() <= to_be_copied <= Rs

    ext_bits = dict(prev_bits)
    for V in Ex.V:
        if V not in ext_bits:
            ext_bits[V] = 1
        else:
            # R1 -> R2 -> Y can be possible, which means R1->R2->Y and R2->Y
            if V not in Ex.sources() and V not in (Ex.sinks() - cannot_merge_sink):
                ext_bits[V] += 1

    if not noiseless:
        for U in new_Vs_Us:
            ext_bits[U] = 1

    assert all(ext_bits[V] == 1 for V in (Ex.sinks() - cannot_merge_sink) | Ex.sources())

    def func_for(V: str, model_id: int):
        def func(vv: Dict[str, int]):
            flipper = 0
            if not noiseless:
                flipper = vv[f'Ex_U_{V}'] if V in new_Vs else 0  # to be noisy

            T_val = Ms[model_id].F[V](vv) if V in T.V else 0
            # irrelevant Ex or just as is which to be transferred.
            if V not in Ex or V in Ex.sources():
                return T_val

            # non-source Ex, needs to carry bits.
            bit = xors(bits_at(vv[P], ext_bits[P] - 1) for P in Ex.pa(V))

            # take if it needs to take the value below
            if V in to_be_copied:
                bit ^= T_val

            bit ^= flipper

            # if there is no T_val, it is okay.
            if V in Ex.sinks() - cannot_merge_sink:
                return T_val ^ bit

            # it is okay whehter it is more than 1 bit or just 1 bit.
            return T_val + (bit << (ext_bits[V] - 1))

        return func

    D = {v: domain_with_bits(b) for v, b in ext_bits.items()}
    assert M1.more_U == M2.more_U

    MU = 0.05  # some random number not to small, not to close to 0.5

    def P_U(vv):
        p_u = M1.P_U(vv)
        if not noiseless:
            num_ones = sum(vv[_] == 1 for _ in new_Vs_Us if _ in vv)
            num_zeros = sum(vv[_] == 0 for _ in new_Vs_Us if _ in vv)
            return p_u * (MU ** num_ones) * ((1 - MU) ** num_zeros)
        else:
            return p_u

    if not noiseless:
        M1_ = SCM(TEx, F={V: func_for(V, 1) for V in TEx.V}, P_U=P_U, D=D, more_U=M1.more_U | new_Vs_Us)
        M2_ = SCM(TEx, F={V: func_for(V, 2) for V in TEx.V}, P_U=P_U, D=D, more_U=M2.more_U | new_Vs_Us)
        return M1_, M2_, ext_bits
    else:
        M1_ = SCM(TEx, F={V: func_for(V, 1) for V in TEx.V}, P_U=P_U, D=D, more_U=M1.more_U)
        M2_ = SCM(TEx, F={V: func_for(V, 2) for V in TEx.V}, P_U=P_U, D=D, more_U=M2.more_U)
        return M1_, M2_, ext_bits


def path_extended_thicket_models(M1: SCM,
                                 M2: SCM,
                                 TEx: CD,
                                 pG: CD,
                                 W0: str,
                                 prev_bits: Dict[str, int],
                                 *,
                                 noiseless=False):
    assert W0 in TEx.sinks()
    assert prev_bits[W0] == 1

    pTEx = TEx + pG

    non_varying = {V for V in sorted(pG.V) if not pG.pa(V) and not pG.UCs(V)} if not noiseless else set()
    non_varying_Us = {f'pG_UU_{V}' for V in non_varying}  # additional noise

    others = (pG.V - non_varying) if not noiseless else set()  # all
    others_Us = {f'pG_UU_{V}' for V in others}  # additional noise

    more_pG_Us = non_varying_Us | others_Us
    assert more_pG_Us.isdisjoint(prev_bits.keys())
    Ms = [None, M1, M2]

    ext_bits = dict(prev_bits)
    # separate mechanism except W0 being merged
    for V in (pG.V - {W0}) | pG.U | more_pG_Us:
        # noinspection PyUnresolvedReferences
        ext_bits[V] = ext_bits.get(V, 0) + 1

    def func_for(V: str, model_id: int):
        def func(vv: Dict[str, int]):
            flipper = vv[f'pG_UU_{V}'] if f'pG_UU_{V}' in more_pG_Us else 0
            masked_vv = {k: masked(v, prev_bits[k]) for k, v in vv.items() if k in prev_bits}
            back_val = Ms[model_id].F[V](masked_vv) if V in TEx.V else 0
            if V in pG.V:
                pg_bit = xors(bits_at(vv[W], ext_bits[W] - 1) for W in pG.pa(V) | pG.UCs(V)) ^ flipper
                if V == W0:
                    assert 0 <= back_val <= 1
                    return back_val ^ pg_bit
                else:
                    return back_val + (pg_bit << (ext_bits[V] - 1))
            else:
                return back_val

        return func

    D = {V: domain_with_bits(b) for V, b in ext_bits.items()}
    assert M1.more_U == M2.more_U

    MU = 0.05  # some random number not to small, not to close to 0.5

    def P_U(vv):
        p_u = M1.P_U(vv)
        num_ones = sum(vv[U] == 1 for U in others_Us if U in vv)
        num_zeros = sum(vv[U] == 0 for U in others_Us if U in vv)
        return p_u * (MU ** num_ones) * ((1 - MU) ** num_zeros)

    M1_ = SCM(pTEx, F={V: func_for(V, 1) for V in pTEx.V}, P_U=P_U, D=D, more_U=M1.more_U | more_pG_Us)
    M2_ = SCM(pTEx, F={V: func_for(V, 2) for V in pTEx.V}, P_U=P_U, D=D, more_U=M2.more_U | more_pG_Us)
    #
    return M1_, M2_, ext_bits


def construct_pG(G: CD, Ws, Xs: AbstractSet[str], Ys: AbstractSet[str], subWs: AbstractSet[str]):
    """ find single variable Y in Ys that is d-connected with subWs """
    for W0 in shuffled(subWs):
        G_wo_Xs_ub_W0 = (G - Xs).underbar(W0)
        for Y0 in filter(lambda Y: not G_wo_Xs_ub_W0.independent(Y, W0, Ws - {W0}), shuffled(Ys)):
            pG = find_a_dpath_graph(G_wo_Xs_ub_W0, W0, Y0, Ws - {W0})
            return pG, Y0, W0

    assert False, f'{G - Xs}, {Ws}, {Ys}, {subWs}'
