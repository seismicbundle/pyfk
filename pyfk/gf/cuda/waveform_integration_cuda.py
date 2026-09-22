# write cuda kernels with numba, parallel for each ibool

import math

import cupy as cp
import numpy as np
from numba import complex64, complex128, cuda, float32, njit
from pyfk.gf.cuda.utils import (cal_cujn, compoundMatrix, cujn, eVector,
                                haskellMatrix, initialG, initialZ, propagateB,
                                propagateG, propagateZ, separatS, sh_ch)
from pyfk.utils.error_message import PyfkError
from pyfk.utils.options import frequency_bin_ceiling, resolve_mixed_precision


def _waveform_integration(
        nfft2: int,
        dw: float,
        pmin: float,
        dk: float,
        kc: float,
        pmax: float,
        receiver_distance: np.ndarray,
        wc1: int,
        vs: np.ndarray,
        vp: np.ndarray,
        qs: np.ndarray,
        qp: np.ndarray,
        flip: bool,
        filter_const: float,
        dynamic: bool,
        wc2: int,
        t0: int,
        src_type: int,
        taper: float,
        wc: int,
        mu: np.ndarray,
        thickness: np.ndarray,
        si: np.ndarray,
        src_layer: int,
        rcv_layer: int,
        updn: int,
        epsilon: float,
        sigma: float,
        sum_waveform: np.ndarray,
        cuda_divide_num: int,
        fmax=None,
        mixed_precision=None):
    # * optional frequency ceiling, as in the CPU path; higher bins stay zero
    nfft2 = frequency_bin_ceiling(nfft2, dw, fmax)
    if nfft2 <= wc1 - 1:
        return
    # * generate kp and ks array ((nfft2-wc1+1)*len(thickness))
    kp_list = np.zeros((nfft2-wc1+1, len(thickness)), dtype=complex)
    ks_list = np.zeros((nfft2-wc1+1, len(thickness)), dtype=complex)

    # * get the n list, kp and ks
    n_list: np.ndarray = np.zeros(nfft2-wc1+1, dtype=int)
    n_list_accumulate: np.ndarray = np.zeros(nfft2-wc1+1, dtype=int)
    get_n_list_kpks(wc1, nfft2, kc, dw, pmin, pmax, dk, sigma, thickness, vp, vs, qp, qs,
                    n_list, n_list_accumulate, kp_list, ks_list)
    n_all: int = int(n_list_accumulate[-1])

    # * tabulate the Bessel functions on the device
    # With pmin == 0 the wavenumber grid does not depend on the frequency (see
    # the CPU module). The tables and sum_waveform stay on the device until
    # all chunks are done. Other pmin values keep the host projection.
    use_table = pmin == 0.0
    if use_table:
        n_max = int(np.max(n_list))
        k_table = (0.5 + np.arange(n_max, dtype=np.float64)) * dk
        z_table = k_table[:, np.newaxis] * \
            np.asarray(receiver_distance, dtype=np.float64)[np.newaxis, :]
        z_table_d = cp.asarray(z_table)
        aj0_table_d = cujn(0, z_table_d)
        aj1_table_d = cujn(1, z_table_d)
        aj2_table_d = cujn(2, z_table_d)
        sum_waveform_d = cuda.to_device(sum_waveform)

    # * optional mixed precision (experimental, off by default)
    # Bins where the FP32 propagator could overflow form a low-frequency band
    # ik < ik_fp64_end that runs in FP64; the other bins run in FP32.
    # pos_fp64_end is the same boundary in the flat position index.
    # Requires the tabulated projection, so pmin != 0 always runs in FP64.
    mixed_fp32 = use_table and resolve_mixed_precision(mixed_precision)
    if mixed_fp32:
        ik_fp64_end = _fp32_overflow_band_end(
            nfft2, wc1, dw, dk, sigma, pmin, thickness, vs, epsilon, n_list)
        # DC can lose accuracy through cancellation in FP32; keep it in FP64
        ik_fp64_end = max(ik_fp64_end, min(nfft2, 1))
        nbin_fp64 = ik_fp64_end - (wc1 - 1)
        pos_fp64_end = int(n_list_accumulate[nbin_fp64 - 1]) \
            if nbin_fp64 > 0 else 0
    else:
        ik_fp64_end = nfft2
        pos_fp64_end = n_all

    flip_val = -1. if flip else 1.

    # * inputs shared by every chunk, uploaded once
    n_list_d = cuda.to_device(n_list)
    n_list_accumulate_d = cuda.to_device(n_list_accumulate)
    kp_list_d = cuda.to_device(kp_list)
    ks_list_d = cuda.to_device(ks_list)
    thickness_d = cuda.to_device(thickness)
    mu_d = cuda.to_device(mu)
    si_d = cuda.to_device(si)

    # same partition as np.array_split(range(n_all), cuda_divide_num),
    # without building the range
    chunk_len = n_all // cuda_divide_num
    chunk_rem = n_all % cuda_divide_num

    for index_cuda_divide in range(cuda_divide_num):
        # * current index_cuda_divide info
        if index_cuda_divide < chunk_rem:
            current_n_all = chunk_len + 1
        else:
            current_n_all = chunk_len
        if current_n_all == 0:
            continue
        current_offset = index_cuda_divide * chunk_len + \
            min(index_cuda_divide, chunk_rem)

        threadsperblock = 128
        blockspergrid = (current_n_all + (threadsperblock - 1)
                         ) // threadsperblock

        # positions are ordered by frequency, so the FP64 part is a prefix
        n_fp64_chunk = min(max(pos_fp64_end - current_offset, 0),
                           current_n_all)
        n_fp32_chunk = current_n_all - n_fp64_chunk

        # * device arrays owned by this chunk: u, ik_list, i_list
        try:
            ik_list_d = cp.zeros(current_n_all, dtype=int)
            i_list_d = cp.zeros(current_n_all, dtype=int)
            u_d = cp.zeros((n_fp64_chunk, 3, 3), dtype=complex)
            u32_d = cp.zeros((n_fp32_chunk, 3, 3), dtype=cp.complex64)
        except cp.cuda.memory.OutOfMemoryError as err:
            raise PyfkError(
                "Not enough GPU memory for this chunk, try to make CUDA_DIVIDE_NUM larger!") from err

        # * generate the ik and i list representing the i wavenumber and i frequency
        fill_vals[blockspergrid, threadsperblock](
            n_list_d, n_list_accumulate_d, ik_list_d, i_list_d, wc1, nfft2, current_n_all, current_offset)

        # * run the cuda kernel function, one launch per precision class
        if n_fp64_chunk > 0:
            blocks_fp64 = (n_fp64_chunk + (threadsperblock - 1)
                           ) // threadsperblock
            parallel_kernel[blocks_fp64, threadsperblock](u_d, ik_list_d, i_list_d, kp_list_d, ks_list_d, thickness_d, mu_d, si_d,
                                                          dw, pmin, dk, src_layer, rcv_layer, updn, src_type, epsilon, wc1, n_fp64_chunk)
        if n_fp32_chunk > 0:
            blocks_fp32 = (n_fp32_chunk + (threadsperblock - 1)
                           ) // threadsperblock
            parallel_kernel_f32[blocks_fp32, threadsperblock](u32_d, ik_list_d[n_fp64_chunk:], i_list_d[n_fp64_chunk:], kp_list_d, ks_list_d,
                                                              thickness_d, mu_d, si_d, dw, pmin, dk, src_layer, rcv_layer, updn, src_type, epsilon, wc1, n_fp32_chunk)

        # * get sum_waveform
        if use_table:
            # position range of each frequency bin, clipped to this chunk
            first_bin = int(np.searchsorted(
                n_list_accumulate, current_offset, side="right"))
            last_bin = int(np.searchsorted(
                n_list_accumulate, current_offset + current_n_all - 1,
                side="right"))
            bin_index = np.arange(first_bin, last_bin + 1, dtype=np.int64)
            bin_ik = bin_index + wc1 - 1
            bin_starts = np.maximum(
                n_list_accumulate[bin_index] - n_list[bin_index]
                - current_offset, 0)
            bin_ends = np.minimum(
                n_list_accumulate[bin_index] - current_offset, current_n_all)
            nrec = len(receiver_distance)

            # pos_fp64_end is a bin boundary, so no bin spans both precisions
            in_fp64 = bin_ik < ik_fp64_end
            n_fp64_bins = int(np.count_nonzero(in_fp64))
            if n_fp64_bins > 0:
                bin_ik_d = cuda.to_device(bin_ik[in_fp64])
                bin_starts_d = cuda.to_device(bin_starts[in_fp64])
                bin_ends_d = cuda.to_device(bin_ends[in_fp64])
                get_sum_waveform_bins[n_fp64_bins, (32, 7)](
                    sum_waveform_d, u_d, i_list_d, z_table_d, aj0_table_d,
                    aj1_table_d, aj2_table_d, bin_ik_d, bin_starts_d,
                    bin_ends_d, flip_val, nrec)
            n_fp32_bins = int(np.count_nonzero(~in_fp64))
            if n_fp32_bins > 0:
                # positions relative to the start of the FP32 part
                bin_ik32_d = cuda.to_device(bin_ik[~in_fp64])
                bin_starts32_d = cuda.to_device(
                    bin_starts[~in_fp64] - n_fp64_chunk)
                bin_ends32_d = cuda.to_device(
                    bin_ends[~in_fp64] - n_fp64_chunk)
                get_sum_waveform_bins_f32u[n_fp32_bins, (32, 7)](
                    sum_waveform_d, u32_d, i_list_d[n_fp64_chunk:], z_table_d,
                    aj0_table_d, aj1_table_d, aj2_table_d, bin_ik32_d,
                    bin_starts32_d, bin_ends32_d, flip_val, nrec)
        else:
            ik_list = cp.asnumpy(ik_list_d)
            i_list = cp.asnumpy(i_list_d)
            u = cp.asnumpy(u_d)
            z_list = np.zeros(
                (current_n_all, len(receiver_distance)), dtype=float)
            get_z_list(z_list, ik_list, i_list, receiver_distance,
                       dw, pmin, dk, current_n_all)
            aj0_list = cal_cujn(0, z_list)
            aj1_list = cal_cujn(1, z_list)
            aj2_list = cal_cujn(2, z_list)
            # it's not appropriate to use cuda here, as it will use large atomic operation.
            get_sum_waveform(sum_waveform, u, ik_list, receiver_distance,  flip_val,
                             z_list, aj0_list, aj1_list, aj2_list, current_n_all)

    if use_table:
        # * copy the projected spectrum back to the host
        sum_waveform_d.copy_to_host(sum_waveform)

    # * perform the filtering
    apply_filter(wc1, nfft2, dw, filter_const, dynamic, wc, taper, wc2,
                 receiver_distance, t0, sum_waveform)


@njit("void(int64,int64,float64,float64,float64,float64,float64,float64,float64[:],float64[:],float64[:],float64[:],float64[:],int64[:],int64[:],complex128[:,:],complex128[:,:])")
def get_n_list_kpks(wc1: int, nfft2: int, kc: float, dw: float, pmin: float, pmax: float, dk: float, sigma: float,
                    thickness: np.ndarray, vp: np.ndarray, vs: np.ndarray, qp: np.ndarray, qs: np.ndarray,
                    # output
                    n_list: np.ndarray, n_list_accumulate: np.ndarray,  kp_list: np.ndarray, ks_list: np.ndarray):
    n_all = 0
    for ik in range(wc1 - 1, nfft2):
        kc2 = kc**2
        omega = ik * dw
        w = omega - sigma * 1j
        att = np.log(w / (2 * np.pi)) / np.pi + 0.5j
        k = omega * pmin + 0.5 * dk
        n = int(((kc2 + (pmax * omega)**2)**0.5 - k) / dk)
        n_list[ik-wc1+1] = n
        n_all = n_all+n
        n_list_accumulate[ik-wc1+1] = n_all
        for idep in range(len(thickness)):
            kp_list[ik-wc1+1, idep] = (w /
                                       (vp[idep] * (1. + att / qp[idep])))**2
            ks_list[ik-wc1+1, idep] = (w /
                                       (vs[idep] * (1. + att / qs[idep])))**2


@cuda.jit("void(int64[:],int64[:],int64[:],int64[:],int64,int64,int64,int64)")
def fill_vals(n_list: np.ndarray, n_list_accumulate: np.ndarray, ik_list: np.ndarray, i_list: np.ndarray, wc1: int, nfft2: int, current_n_all: int, current_offset: int):
    pos = cuda.grid(1)
    if pos < current_n_all:
        ibool = pos+current_offset
        # first index with n_list_accumulate[index] > ibool; the array is
        # nondecreasing, so the binary search lands on the same index a
        # linear scan would find, and ibool < n_list_accumulate[-1]
        # guarantees it exists
        lo = 0
        hi = nfft2 - wc1
        while lo < hi:
            mid = (lo + hi) // 2
            if n_list_accumulate[mid] > ibool:
                hi = mid
            else:
                lo = mid + 1
        ik_list[pos] = lo + wc1 - 1
        i_list[pos] = ibool - (n_list_accumulate[lo] - n_list[lo])


@njit("void(float64[:,:],int64[:],int64[:],float64[:],float64,float64,float64,int64)")
def get_z_list(z_list, ik_list, i_list, receiver_distance, dw, pmin, dk, current_n_all):
    for pos in range(current_n_all):
        ik = ik_list[pos]
        i = i_list[pos]
        omega = ik * dw
        k = omega * pmin + (0.5+i) * dk
        for irec in range(len(receiver_distance)):
            z_list[pos, irec] = k * receiver_distance[irec]


@njit("void(complex128[:,:,:],complex128[:,:,:],int64[:],float64[:],float64,float64[:,:],float64[:,:],float64[:,:],float64[:,:],int64)")
def get_sum_waveform(sum_waveform, u, ik_list, receiver_distance,  flip_val,
                     z_list, aj0_list, aj1_list, aj2_list, current_n_all):
    for pos in range(current_n_all):
        ik = ik_list[pos]
        for irec in range(len(receiver_distance)):
            z = z_list[pos, irec]
            aj0 = aj0_list[pos, irec]
            aj1 = aj1_list[pos, irec]
            aj2 = aj2_list[pos, irec]
            # do the numerical integration here
            sum_waveform[irec, 0, ik] += u[pos, 0, 0] * \
                aj0 * flip_val
            sum_waveform[irec, 1, ik] += -u[pos, 0, 1] * aj1
            sum_waveform[irec, 2, ik] += -u[pos, 0, 2] * aj1

            nf = (u[pos, 1, 1] + u[pos, 1, 2]) * aj1 / z
            sum_waveform[irec, 3, ik] += u[pos, 1, 0] * \
                aj1 * flip_val
            sum_waveform[irec, 4, ik] += u[pos, 1, 1] * aj0 - nf
            sum_waveform[irec, 5, ik] += u[pos, 1, 2] * aj0 - nf

            nf = 2. * (u[pos, 2, 1] + u[pos, 2, 2]) * aj2 / z
            sum_waveform[irec, 6, ik] += u[pos, 2, 0] * \
                aj2 * flip_val
            sum_waveform[irec, 7, ik] += u[pos, 2, 1] * aj1 - nf
            sum_waveform[irec, 8, ik] += u[pos, 2, 2] * aj1 - nf


@cuda.jit("void(complex128[:,:,:],complex128[:,:,:],int64[:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],int64[:],int64[:],int64[:],float64,int64)")
def get_sum_waveform_bins(sum_waveform, u, i_list, z_table, aj0_table,
                          aj1_table, aj2_table, bin_ik, bin_starts, bin_ends,
                          flip_val, nrec):
    # One block per frequency bin, one warp per component group (0, 1, 2, 3,
    # 4+5, 6, 7+8), one lane per receiver. Each output element is summed by
    # one thread in the order of the host loop, continuing from the stored
    # value when a bin is split across chunks.
    ibin = cuda.blockIdx.x
    start = bin_starts[ibin]
    end = bin_ends[ibin]
    if start < end:
        ik = bin_ik[ibin]
        group = cuda.threadIdx.y
        for irec in range(cuda.threadIdx.x, nrec, 32):
            if group == 0:
                acc = sum_waveform[irec, 0, ik]
                for pos in range(start, end):
                    aj0 = aj0_table[i_list[pos], irec]
                    acc = acc + u[pos, 0, 0] * aj0 * flip_val
                sum_waveform[irec, 0, ik] = acc
            elif group == 1:
                acc = sum_waveform[irec, 1, ik]
                for pos in range(start, end):
                    aj1 = aj1_table[i_list[pos], irec]
                    acc = acc + -u[pos, 0, 1] * aj1
                sum_waveform[irec, 1, ik] = acc
            elif group == 2:
                acc = sum_waveform[irec, 2, ik]
                for pos in range(start, end):
                    aj1 = aj1_table[i_list[pos], irec]
                    acc = acc + -u[pos, 0, 2] * aj1
                sum_waveform[irec, 2, ik] = acc
            elif group == 3:
                acc = sum_waveform[irec, 3, ik]
                for pos in range(start, end):
                    aj1 = aj1_table[i_list[pos], irec]
                    acc = acc + u[pos, 1, 0] * aj1 * flip_val
                sum_waveform[irec, 3, ik] = acc
            elif group == 4:
                acc4 = sum_waveform[irec, 4, ik]
                acc5 = sum_waveform[irec, 5, ik]
                for pos in range(start, end):
                    i = i_list[pos]
                    z = z_table[i, irec]
                    aj0 = aj0_table[i, irec]
                    aj1 = aj1_table[i, irec]
                    nf = (u[pos, 1, 1] + u[pos, 1, 2]) * aj1 / z
                    acc4 = acc4 + (u[pos, 1, 1] * aj0 - nf)
                    acc5 = acc5 + (u[pos, 1, 2] * aj0 - nf)
                sum_waveform[irec, 4, ik] = acc4
                sum_waveform[irec, 5, ik] = acc5
            elif group == 5:
                acc = sum_waveform[irec, 6, ik]
                for pos in range(start, end):
                    aj2 = aj2_table[i_list[pos], irec]
                    acc = acc + u[pos, 2, 0] * aj2 * flip_val
                sum_waveform[irec, 6, ik] = acc
            else:
                acc7 = sum_waveform[irec, 7, ik]
                acc8 = sum_waveform[irec, 8, ik]
                for pos in range(start, end):
                    i = i_list[pos]
                    z = z_table[i, irec]
                    aj1 = aj1_table[i, irec]
                    aj2 = aj2_table[i, irec]
                    nf = 2. * (u[pos, 2, 1] + u[pos, 2, 2]) * aj2 / z
                    acc7 = acc7 + (u[pos, 2, 1] * aj1 - nf)
                    acc8 = acc8 + (u[pos, 2, 2] * aj1 - nf)
                sum_waveform[irec, 7, ik] = acc7
                sum_waveform[irec, 8, ik] = acc8


# Disable fused multiply-add contraction in the projection kernel so it rounds
# like the host code. Numba has no public option for this, so the private
# NVVM options are set and the compiled caches are cleared. The check below
# fails at import if the generated code still contains FP64 FMA instructions.
for _definition in get_sum_waveform_bins.overloads.values():
    _lib = _definition.library
    _lib._nvvm_options["fma"] = False
    for _name in ("_ptx_cache", "_cubin_cache", "_cufunc_cache"):
        _cur = getattr(_lib, _name, None)
        if isinstance(_cur, dict):
            _cur.clear()
        else:
            setattr(_lib, _name, None)
_asm = "\n".join(str(_a) for _a in get_sum_waveform_bins.inspect_asm().values())
if "fma.rn.f64" in _asm:
    raise RuntimeError("CUDA projection contraction detected; install the pinned pyfk[cuda] dependencies")
del _definition, _lib, _asm


@njit("void(int64,int64,float64,float64,boolean,int64,float64,int64,float64[:],float64[:],complex128[:,:,:])")
def apply_filter(wc1, nfft2, dw, filter_const, dynamic, wc, taper, wc2,
                 receiver_distance, t0, sum_waveform):
    for ik in range(wc1 - 1, nfft2):
        omega = ik * dw
        # * for each ik, we apply the filtering and apply the time shift in the frequency domain
        filtering = filter_const
        if dynamic and (ik + 1 > wc):
            filtering = 0.5 * (1. + np.cos((ik + 1 - wc) * taper)) * filtering
        if dynamic and (ik + 1 < wc2):
            filtering = 0.5 * \
                (1. + np.cos((wc2 - ik - 1) * np.pi / (wc2 - wc1))) * filtering
        # in fk's code, only apply atttemp for ncom, here we apply to all, with
        # no difference
        for icom in range(9):
            for irec in range(len(receiver_distance)):
                phi = omega * t0[irec]
                atttemp = filtering * (np.cos(phi) + np.sin(phi) * 1j)
                sum_waveform[irec, icom,
                             ik] = sum_waveform[irec, icom, ik]*atttemp


@cuda.jit("void(complex128[:,:,:],int64[:],int64[:],complex128[:,:],complex128[:,:],float64[:],float64[:],float64[:,:],float64,float64,float64,int64,int64,int64,int64,float64,int64,int64)")
def parallel_kernel(u, ik_list, i_list, kp_list, ks_list, thickness, mu, si,
                    dw, pmin, dk, src_layer, rcv_layer, updn, src_type, epsilon, wc1, current_n_all):
    # * pos for this thread
    pos = cuda.grid(1)
    if(pos < current_n_all):
        ik = ik_list[pos]
        i = i_list[pos]

        # * generate some local arrays
        aaa = cuda.local.array((5, 5), dtype=complex128)
        bbb = cuda.local.array((7, 7), dtype=complex128)
        ccc = cuda.local.array((7, 7), dtype=complex128)
        eee = cuda.local.array(7, dtype=complex128)
        ggg = cuda.local.array(7, dtype=complex128)
        zzz = cuda.local.array((3, 5), dtype=complex128)
        sss = cuda.local.array((3, 6), dtype=complex128)
        temppp = cuda.local.array((4, 4), dtype=complex128)

        omega = ik * dw
        k = omega * pmin + (0.5+i) * dk

        # * here we directly run the kernel function as in cython
        aaa[:, :] = 0. + 0.j
        bbb[:, :] = 0. + 0.j
        ccc[:, :] = 0. + 0.j
        eee[:] = 0. + 0.j
        ggg[:] = 0. + 0.j
        zzz[:, :] = 0. + 0.j
        sss[:, :] = 0. + 0.j
        temppp[:, :] = 0. + 0.j

        for index in range(7):
            bbb[index, index] = 1.
        eee[0], eee[5], ggg[4], ggg[6] = 1., 1., 1., 1.

        for ilayer in range(len(thickness) - 1, -1, -1):
            Ca, Ya, Xa, Cb, Yb, Xb = 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j
            exa, exb = 0., 0.
            kka = kp_list[ik-wc1+1, ilayer] / (k**2)
            kkb = ks_list[ik-wc1+1, ilayer] / (k**2)
            r = 2. / kkb
            kd = k * thickness[ilayer]
            mu2 = 2. * mu[ilayer]
            ra = (1. - kka)**0.5
            rb = (1. - kkb)**0.5
            r1 = 1. - 1. / r

            if ilayer == len(thickness) - 1 and thickness[ilayer] < epsilon:
                # * begin initialG
                # ? Initialize the g row-vector. The first 5 elements are the
                # ? inverse(E)|_{ij}^{12}, ij=12,13,23,24,34.
                # ? g14 is omitted because g14 = -g23.
                # ? The last two are the 5th row of E^-1.
                # p-sv, see EQ 33 on ZR/p623, constant omitted.
                initialG(ggg, r, ra, rb, r1, mu2)
                # * end initailG
            elif ilayer == 0 and thickness[ilayer] < epsilon:
                # * begin eVector(e)
                # ? The first 5 members are E|_12^ij, ij=12,13,23,24,34.
                # ? The last two are the first column of SH E matrix.
                # For p-sv, compute E|_(12)^(ij), ij=12, 13, 23, 24, 34.
                eVector(eee, ra, rb, r1, mu2)
                # * end eVector(e)
                break
            else:
                # * begin compoundMatrix(c)
                # ? The upper-left 5x5 is the 6x6 compound matrix of the P-SV Haskell matrix,
                # ?       a(ij,kl) = A|_kl^ij, ij=12,13,14,23,24,34,
                # ? after dropping the 3rd row and colummn and replacing the 4th row
                # ? by (2A41, 2A42, 2A44-1,2A45,2A46) (see W&H, P1035).
                # ? The lower-right c 2x2 is the SH part of the Haskell matrix.
                # ? Input: layer parameters.
                # ? Output: compound matrix a, scaled by exa*exb for the P-SV and exb for the SH.
                Ca, Ya, Xa, exa = sh_ch(ra, kd)
                Cb, Yb, Xb, exb = sh_ch(rb, kd)

                compoundMatrix(ccc, Ca, Ya, Xa, Cb, Yb, Xb,
                               exa, exb, r, r1,  mu2)
                # * end compoundMatrix(c)

                # * begin propagateG(c, g)
                # ? propagate g vector upward using the compound matrix
                # ?       g = g*a
                propagateG(ggg, ccc)
                # * end propagateG(c, g)
            if ilayer == src_layer:
                # * begin separatS
                separatS(sss, updn, src_type, ra, rb, r, mu2, temppp, r1, si)
                # * end separatS

                # * begin initialZ(ss, g, z)
                # ? initialize the row-vector z at the source z(j)=s(i)*X|_ij^12
                # ? for P-SV and z(j)=s(i)*X(5,i) for SH.
                # ?  input:
                # ?       s(3,6)  ---- source coef. for n=0,1,2
                # ?       g(7)    ---- g vector used to construct matrix X|_ij^12
                # ?                    |  0   g1  g2 -g3 |
                # ?        X|_ij^12 =  | -g1  0   g3  g4 | for P-SV.
                # ?                    | -g2 -g3  0   g5 |
                # ?                    |  g3 -g4 -g5  0  |
                # ?        X(5,i) = ( g6 g7 )     for SH.
                # ?  output:
                # ?       z(3,5)  ---- z vector for n=0,1,2
                initialZ(zzz, ggg, sss)
                # * end initialZ(ss, g, z)
            if ilayer < src_layer:
                if ilayer >= rcv_layer:
                    # * begin haskellMatrix(a)
                    # ? compute 4x4 P-SV Haskell a for the layer
                    haskellMatrix(aaa, Ca, Ya, Xa, Cb, Yb,
                                  Xb, exa, exb, r, r1, mu2)
                    # * end haskellMatrix(a)

                    # * begin propagateZ(a, z)
                    propagateZ(zzz, aaa)
                    # * end propagateZ(a, z)
                else:
                    # * begin propagateB(c, b)

                    propagateB(bbb, ccc)
                    # * end propagateB(c, b)
        # c add the top halfspace boundary condition

        rayl, love = 0. + 0.j, 0. + 0.j
        for index in range(5):
            rayl = rayl + ggg[index] * eee[index]
        for index in range(5, 7):
            love = love + ggg[index] * eee[index]
        for index in range(4):
            ggg[index] = 0. + 0.j
        for imat in range(4):
            for jmat in range(5):
                ggg[imat] = ggg[imat] + bbb[imat, jmat] * eee[jmat]
        ggg[2] /= 2.
        ggg[5] = bbb[5, 5] * eee[5] + bbb[5, 6] * eee[6]
        for index in range(3):
            val = zzz[index, 1] * ggg[0] + zzz[index, 2] * \
                ggg[1] - zzz[index, 3] * ggg[2]
            zzz[index, 1] = -zzz[index, 0] * ggg[0] + \
                zzz[index, 2] * ggg[2] + zzz[index, 3] * ggg[3]
            zzz[index, 0] = val
            zzz[index, 4] = zzz[index, 4] * ggg[5]
        val = k
        if src_type == 1:
            val = 1.

        # * set u
        for index in range(3):
            u[pos, index, 0] = u[pos, index, 0] + val * zzz[index, 1] / rayl
            u[pos, index, 1] = u[pos, index, 1] + val * zzz[index, 0] / rayl
            u[pos, index, 2] = u[pos, index, 2] + val * zzz[index, 4] / love


def _fp32_overflow_band_end(nfft2, wc1, dw, dk, sigma, pmin, thickness, vs,
                            epsilon, n_list):
    """First frequency bin index that may run the FP32 propagator.

    Per propagated layer, the compound-matrix factors are bounded by
    (2 k^2 vs^2 / |w|^2)^2, largest at the top of the bin's wavenumber grid.
    The estimated growth E = sum over layers of max(0, log2(2^-24 * 2 * bound^2))
    must stay below the FP32 exponent limit 128 minus a 48 bit margin
    (36 bits for one extra layer factor inside propagateG, 12 bits for the
    final assembly). Halfspaces are excluded. E decreases with frequency, and
    the returned index follows the last bin that fails the test.
    """
    eps32 = 2.0 ** -24
    threshold = 128.0 - 48.0
    nlayer = len(thickness)
    chain_layers = [ilayer for ilayer in range(nlayer)
                    if not (ilayer == nlayer - 1
                            and thickness[ilayer] < epsilon)
                    and not (ilayer == 0 and thickness[ilayer] < epsilon)]
    ik = np.arange(wc1 - 1, nfft2, dtype=np.float64)
    omega = ik * dw
    w2 = omega * omega + sigma * sigma
    # top of each bin's k grid, from the same n the wavenumber loop uses
    ktop = omega * pmin + (0.5 + np.asarray(n_list, dtype=np.float64)
                           - 1.0) * dk
    growth = np.zeros(nfft2 - wc1 + 1, dtype=np.float64)
    for ilayer in chain_layers:
        r_bound = 2.0 * ktop * ktop * (vs[ilayer] * vs[ilayer]) / w2
        growth += np.maximum(0.0, np.log2(eps32 * 2.0 * r_bound * r_bound))
    flagged = np.nonzero(growth >= threshold)[0]
    if flagged.size == 0:
        return wc1 - 1
    return int(flagged[-1]) + wc1


# FP32 versions of the device functions in pyfk.gf.cuda.utils. Literals are
# float32 so that no operation is promoted to double.


@cuda.jit("complex64(complex64)", device=True)
def csqrt_f32(z):
    # principal square root; the branch avoids cancellation when re < 0
    re = z.real
    im = z.imag
    mag = math.sqrt(re * re + im * im)
    if re >= float32(0.):
        w = math.sqrt(float32(0.5) * (mag + re))
        return complex64(complex(w, im / (float32(2.) * w)))
    w = math.sqrt(float32(0.5) * (mag - re))
    u = abs(im) / (float32(2.) * w)
    if im < float32(0.):
        return complex64(complex(u, -w))
    return complex64(complex(u, w))


@cuda.jit("Tuple((complex64,complex64,complex64,float32))(complex64,float32)", device=True)
def sh_ch_f32(a, kd):
    y = kd * a
    r = y.real
    i = y.imag
    ex = math.exp(-r)
    y = float32(0.5) * complex64(complex(math.cos(i), math.sin(i)))
    x = ex * ex * y.conjugate()
    c = y + x
    x = y - x
    y = x / a
    x = x * a
    return c, y, x, ex


@cuda.jit("void(complex64[:],complex64,complex64,complex64,complex64,float32)", device=True)
def initialG_f32(ggg, r, ra, rb, r1, mu2):
    delta = r * (float32(1.) - ra * rb) - float32(1.)
    ggg[0] = mu2 * (delta - r1)
    ggg[1] = ra
    ggg[2] = delta
    ggg[3] = -rb
    ggg[4] = (float32(1.) + delta) / mu2
    ggg[5] = float32(-1.)
    ggg[6] = float32(2.) / (rb * mu2)


@cuda.jit("void(complex64[:],complex64,complex64,complex64,float32)", device=True)
def eVector_f32(eee, ra, rb, r1, mu2):
    eee[0] = ra * rb - float32(1.)
    eee[1] = mu2 * rb * (float32(1.) - r1)
    eee[2] = mu2 * (r1 - ra * rb)
    eee[3] = mu2 * ra * (r1 - float32(1.))
    eee[4] = mu2 * mu2 * (ra * rb - r1 * r1)
    eee[5] = float32(-1.)
    eee[6] = mu2 * rb / float32(2.)


@cuda.jit("void(complex64[:,:],complex64,complex64,complex64,complex64,complex64,complex64,float32,float32,complex64,complex64,float32)", device=True)
def compoundMatrix_f32(ccc, Ca, Ya, Xa, Cb, Yb, Xb, exa, exb, r, r1, mu2):
    CaCb = Ca * Cb
    CaYb = Ca * Yb
    CaXb = Ca * Xb
    XaCb = Xa * Cb
    XaXb = Xa * Xb
    YaCb = Ya * Cb
    YaYb = Ya * Yb
    ex = exa * exb
    r2 = r * r
    r3 = r1 * r1

    # p-sv, scaled by exa*exb to suppress overflow
    ccc[0, 0] = ((float32(1.) + r3) * CaCb - XaXb -
                 r3 * YaYb - float32(2.) * r1 * ex) * r2
    ccc[0, 1] = (XaCb - CaYb) * r / mu2
    ccc[0, 2] = ((float32(1.) + r1) * (CaCb - ex) -
                 XaXb - r1 * YaYb) * r2 / mu2
    ccc[0, 3] = (YaCb - CaXb) * r / mu2
    ccc[0, 4] = (float32(2.) * (CaCb - ex) - XaXb - YaYb) * r2 / (mu2 * mu2)

    ccc[1, 0] = (r3 * YaCb - CaXb) * r * mu2
    ccc[1, 1] = CaCb
    ccc[1, 2] = (r1 * YaCb - CaXb) * r
    ccc[1, 3] = -Ya * Xb
    ccc[1, 4] = ccc[0, 3]

    ccc[2, 0] = float32(2.) * mu2 * r2 * \
        (r1 * r3 * YaYb - (CaCb - ex) * (r3 + r1) + XaXb)
    ccc[2, 1] = float32(2.) * r * (r1 * CaYb - XaCb)
    ccc[2, 2] = float32(2.) * (CaCb - ccc[0, 0]) + ex
    ccc[2, 3] = float32(-2.) * ccc[1, 2]
    ccc[2, 4] = float32(-2.) * ccc[0, 2]

    ccc[3, 0] = mu2 * r * (XaCb - r3 * CaYb)
    ccc[3, 1] = -Xa * Yb
    ccc[3, 2] = -ccc[2, 1] / float32(2.)
    ccc[3, 3] = ccc[1, 1]
    ccc[3, 4] = ccc[0, 1]

    ccc[4, 0] = mu2 * mu2 * r2 * \
        (float32(2.) * (CaCb - ex) * r3 - XaXb - r3 * r3 * YaYb)
    ccc[4, 1] = ccc[3, 0]
    ccc[4, 2] = -ccc[2, 0] / float32(2.)
    ccc[4, 3] = ccc[1, 0]
    ccc[4, 4] = ccc[0, 0]

    # sh, scaled by exb
    ccc[5, 5] = Cb
    ccc[5, 6] = float32(-2.) * Yb / mu2
    ccc[6, 5] = -mu2 * Xb / float32(2.)
    ccc[6, 6] = Cb


@cuda.jit("void(complex64[:],complex64[:,:])", device=True)
def propagateG_f32(ggg, ccc):
    ggg_temp = cuda.local.array(7, dtype=complex64)
    ggg_temp[:] = 0. + 0.j
    for imat in range(7):
        for jmat in range(7):
            ggg_temp[imat] = ggg_temp[imat] + \
                ggg[jmat] * ccc[jmat, imat]

    for imat in range(7):
        ggg[imat] = ggg_temp[imat]


@cuda.jit("void(complex64[:,:],int64,int64,complex64,complex64,complex64,float32,complex64[:,:],complex64,float64[:,:])", device=True)
def separatS_f32(sss, updn, src_type, ra, rb, r, mu2, temppp, r1, si):
    # si is float64 and is converted to float32 when read
    if updn == 0:
        if src_type == 2:
            for index in range(6):
                for isrc_type in range(3):
                    sss[isrc_type, index] = si[isrc_type, index]
        elif src_type == 1:
            for index in range(6):
                for isrc_type in range(2):
                    sss[isrc_type, index] = si[isrc_type, index]
        else:
            for index in range(6):
                for isrc_type in range(1):
                    sss[isrc_type, index] = si[isrc_type, index]
    else:
        ra1 = float32(1.) / ra
        rb1 = float32(1.) / rb
        if updn == 1:
            dum = -r
            temp_sh = (float32(-2.) / mu2) * rb1
        else:
            dum = r
            temp_sh = (float32(2.) / mu2) * rb1

        temppp[0, 0] = float32(1.)
        temppp[0, 1] = dum * (rb - r1 * ra1)
        temppp[0, 2] = float32(0.)
        temppp[0, 3] = dum * (ra1 - rb) / mu2
        temppp[1, 0] = dum * (ra - r1 * rb1)
        temppp[1, 1] = float32(1.)
        temppp[1, 2] = dum * (rb1 - ra) / mu2
        temppp[1, 3] = float32(0.)
        temppp[2, 0] = float32(0.)
        temppp[2, 1] = dum * (rb - r1 * r1 * ra1) * mu2
        temppp[2, 2] = float32(1.)
        temppp[2, 3] = dum * (r1 * ra1 - rb)
        temppp[3, 0] = dum * (ra - r1 * r1 * rb1) * mu2
        temppp[3, 1] = float32(0.)
        temppp[3, 2] = dum * (r1 * rb1 - ra)
        temppp[3, 3] = float32(1.)

        for imat in range(src_type + 1):
            for kmat in range(4):
                ctemp = complex64(0j)
                for jmat in range(4):
                    ctemp = ctemp + float32(si[imat, jmat]) * \
                        temppp[kmat, jmat]
                sss[imat, kmat] = ctemp / float32(2.)
            sss[imat, 4] = (float32(si[imat, 4])
                            + temp_sh * float32(si[imat, 5])) / float32(2.)
            sss[imat, 5] = (float32(si[imat, 5])
                            + float32(si[imat, 4]) / temp_sh) / float32(2.)


@cuda.jit("void(complex64[:,:],complex64[:],complex64[:,:])", device=True)
def initialZ_f32(zzz, ggg, sss):
    for index in range(3):
        # p-sv
        zzz[index, 0] = -sss[index, 1] * ggg[0] - \
            sss[index, 2] * ggg[1] + sss[index, 3] * ggg[2]
        zzz[index, 1] = sss[index, 0] * ggg[0] - \
            sss[index, 2] * ggg[2] - sss[index, 3] * ggg[3]
        zzz[index, 2] = sss[index, 0] * ggg[1] + \
            sss[index, 1] * ggg[2] - sss[index, 3] * ggg[4]
        zzz[index, 3] = -sss[index, 0] * ggg[2] + \
            sss[index, 1] * ggg[3] + sss[index, 2] * ggg[4]
        # sh
        zzz[index, 4] = sss[index, 4] * ggg[5] + sss[index, 5] * ggg[6]


@cuda.jit("void(complex64[:,:],complex64,complex64,complex64,complex64,complex64,complex64,float32,float32,complex64,complex64,float32)", device=True)
def haskellMatrix_f32(aaa, Ca, Ya, Xa, Cb, Yb, Xb, exa, exb, r, r1, mu2):
    Ca = Ca * exb
    Xa = Xa * exb
    Ya = Ya * exb
    Cb = Cb * exa
    Yb = Yb * exa
    Xb = Xb * exa
    # p-sv, scaled by exa*exb
    aaa[0, 0] = r * (Ca - r1 * Cb)
    aaa[0, 1] = r * (r1 * Ya - Xb)
    aaa[0, 2] = (Cb - Ca) * r / mu2
    aaa[0, 3] = (Xb - Ya) * r / mu2

    aaa[1, 0] = r * (r1 * Yb - Xa)
    aaa[1, 1] = r * (Cb - r1 * Ca)
    aaa[1, 2] = (Xa - Yb) * r / mu2
    aaa[1, 3] = -aaa[0, 2]

    aaa[2, 0] = mu2 * r * r1 * (Ca - Cb)
    aaa[2, 1] = mu2 * r * (r1 * r1 * Ya - Xb)
    aaa[2, 2] = aaa[1, 1]
    aaa[2, 3] = -aaa[0, 1]

    aaa[3, 0] = mu2 * r * (r1 * r1 * Yb - Xa)
    aaa[3, 1] = -aaa[2, 0]
    aaa[3, 2] = -aaa[1, 0]
    aaa[3, 3] = aaa[0, 0]

    # sh, replaced by exb
    aaa[4, 4] = exb


@cuda.jit("void(complex64[:,:],complex64[:,:])", device=True)
def propagateZ_f32(zzz, aaa):
    zzz_temp = cuda.local.array((3, 5), dtype=complex64)
    zzz_temp[:, :] = 0. + 0.j
    for imat in range(3):
        for jmat in range(5):
            for kmat in range(5):
                zzz_temp[imat, kmat] = zzz_temp[imat,
                                                kmat] + zzz[imat, jmat] * aaa[jmat, kmat]

    for imat in range(3):
        for kmat in range(5):
            zzz[imat, kmat] = zzz_temp[imat, kmat]


@cuda.jit("void(complex64[:,:],complex64[:,:])", device=True)
def propagateB_f32(bbb, ccc):
    bbb_temp = cuda.local.array((7, 7), dtype=complex64)

    bbb_temp[:, :] = 0. + 0.j
    for imat in range(7):
        for jmat in range(7):
            for kmat in range(7):
                bbb_temp[imat, kmat] = bbb_temp[imat,
                                                kmat] + bbb[imat, jmat] * ccc[jmat, kmat]

    for imat in range(7):
        for kmat in range(7):
            bbb[imat, kmat] = bbb_temp[imat, kmat]


@cuda.jit("void(complex64[:,:,:],int64[:],int64[:],complex128[:,:],complex128[:,:],float64[:],float64[:],float64[:,:],float64,float64,float64,int64,int64,int64,int64,float64,int64,int64)")
def parallel_kernel_f32(u, ik_list, i_list, kp_list, ks_list, thickness, mu, si,
                        dw, pmin, dk, src_layer, rcv_layer, updn, src_type, epsilon, wc1, current_n_all):
    # parallel_kernel with the layer propagation in FP32. The wavenumber
    # grid, the halfspace tests and the final rayl/love assembly stay FP64.
    pos = cuda.grid(1)
    if(pos < current_n_all):
        ik = ik_list[pos]
        i = i_list[pos]

        # * generate some local arrays
        aaa = cuda.local.array((5, 5), dtype=complex64)
        bbb = cuda.local.array((7, 7), dtype=complex64)
        ccc = cuda.local.array((7, 7), dtype=complex64)
        eee = cuda.local.array(7, dtype=complex64)
        ggg = cuda.local.array(7, dtype=complex64)
        zzz = cuda.local.array((3, 5), dtype=complex64)
        sss = cuda.local.array((3, 6), dtype=complex64)
        temppp = cuda.local.array((4, 4), dtype=complex64)

        omega = ik * dw
        k = omega * pmin + (0.5 + i) * dk
        kf = float32(k)

        aaa[:, :] = 0. + 0.j
        bbb[:, :] = 0. + 0.j
        ccc[:, :] = 0. + 0.j
        eee[:] = 0. + 0.j
        ggg[:] = 0. + 0.j
        zzz[:, :] = 0. + 0.j
        sss[:, :] = 0. + 0.j
        temppp[:, :] = 0. + 0.j

        for index in range(7):
            bbb[index, index] = 1.
        eee[0], eee[5], ggg[4], ggg[6] = 1., 1., 1., 1.

        Ca = complex64(0j)
        Ya = complex64(0j)
        Xa = complex64(0j)
        Cb = complex64(0j)
        Yb = complex64(0j)
        Xb = complex64(0j)
        exa = float32(0.)
        exb = float32(0.)
        for ilayer in range(len(thickness) - 1, -1, -1):
            kka = complex64(kp_list[ik - wc1 + 1, ilayer]) / (kf * kf)
            kkb = complex64(ks_list[ik - wc1 + 1, ilayer]) / (kf * kf)
            r = float32(2.) / kkb
            kd = kf * float32(thickness[ilayer])
            mu2 = float32(2.) * float32(mu[ilayer])
            ra = csqrt_f32(float32(1.) - kka)
            rb = csqrt_f32(float32(1.) - kkb)
            r1 = float32(1.) - float32(1.) / r

            if ilayer == len(thickness) - 1 and thickness[ilayer] < epsilon:
                initialG_f32(ggg, r, ra, rb, r1, mu2)
            elif ilayer == 0 and thickness[ilayer] < epsilon:
                eVector_f32(eee, ra, rb, r1, mu2)
                break
            else:
                Ca, Ya, Xa, exa = sh_ch_f32(ra, kd)
                Cb, Yb, Xb, exb = sh_ch_f32(rb, kd)
                compoundMatrix_f32(ccc, Ca, Ya, Xa, Cb, Yb, Xb,
                                   exa, exb, r, r1, mu2)
                propagateG_f32(ggg, ccc)
            if ilayer == src_layer:
                separatS_f32(sss, updn, src_type, ra,
                             rb, r, mu2, temppp, r1, si)
                initialZ_f32(zzz, ggg, sss)
            if ilayer < src_layer:
                if ilayer >= rcv_layer:
                    haskellMatrix_f32(aaa, Ca, Ya, Xa, Cb, Yb,
                                      Xb, exa, exb, r, r1, mu2)
                    propagateZ_f32(zzz, aaa)
                else:
                    propagateB_f32(bbb, ccc)

        # final assembly in FP64
        rayl, love = 0. + 0.j, 0. + 0.j
        for index in range(5):
            rayl = rayl + complex128(ggg[index]) * complex128(eee[index])
        for index in range(5, 7):
            love = love + complex128(ggg[index]) * complex128(eee[index])
        gggd = cuda.local.array(7, dtype=complex128)
        gggd[:] = 0. + 0.j
        for imat in range(4):
            for jmat in range(5):
                gggd[imat] = gggd[imat] + \
                    complex128(bbb[imat, jmat]) * complex128(eee[jmat])
        gggd[2] /= 2.
        gggd[5] = complex128(bbb[5, 5]) * complex128(eee[5]) + \
            complex128(bbb[5, 6]) * complex128(eee[6])
        zzzd = cuda.local.array((3, 5), dtype=complex128)
        for index in range(3):
            for jmat in range(5):
                zzzd[index, jmat] = complex128(zzz[index, jmat])
        for index in range(3):
            val = zzzd[index, 1] * gggd[0] + zzzd[index, 2] * \
                gggd[1] - zzzd[index, 3] * gggd[2]
            zzzd[index, 1] = -zzzd[index, 0] * gggd[0] + \
                zzzd[index, 2] * gggd[2] + zzzd[index, 3] * gggd[3]
            zzzd[index, 0] = val
            zzzd[index, 4] = zzzd[index, 4] * gggd[5]
        val = k
        if src_type == 1:
            val = 1.

        # * set u (stored as complex64)
        for index in range(3):
            u[pos, index, 0] = u[pos, index, 0] + val * zzzd[index, 1] / rayl
            u[pos, index, 1] = u[pos, index, 1] + val * zzzd[index, 0] / rayl
            u[pos, index, 2] = u[pos, index, 2] + val * zzzd[index, 4] / love


@cuda.jit("void(complex128[:,:,:],complex64[:,:,:],int64[:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],int64[:],int64[:],int64[:],float64,int64)")
def get_sum_waveform_bins_f32u(sum_waveform, u, i_list, z_table, aj0_table,
                               aj1_table, aj2_table, bin_ik, bin_starts,
                               bin_ends, flip_val, nrec):
    # get_sum_waveform_bins for complex64 u; the sums are complex128.
    ibin = cuda.blockIdx.x
    start = bin_starts[ibin]
    end = bin_ends[ibin]
    if start < end:
        ik = bin_ik[ibin]
        group = cuda.threadIdx.y
        for irec in range(cuda.threadIdx.x, nrec, 32):
            if group == 0:
                acc = sum_waveform[irec, 0, ik]
                for pos in range(start, end):
                    aj0 = aj0_table[i_list[pos], irec]
                    acc = acc + complex128(u[pos, 0, 0]) * aj0 * flip_val
                sum_waveform[irec, 0, ik] = acc
            elif group == 1:
                acc = sum_waveform[irec, 1, ik]
                for pos in range(start, end):
                    aj1 = aj1_table[i_list[pos], irec]
                    acc = acc + -complex128(u[pos, 0, 1]) * aj1
                sum_waveform[irec, 1, ik] = acc
            elif group == 2:
                acc = sum_waveform[irec, 2, ik]
                for pos in range(start, end):
                    aj1 = aj1_table[i_list[pos], irec]
                    acc = acc + -complex128(u[pos, 0, 2]) * aj1
                sum_waveform[irec, 2, ik] = acc
            elif group == 3:
                acc = sum_waveform[irec, 3, ik]
                for pos in range(start, end):
                    aj1 = aj1_table[i_list[pos], irec]
                    acc = acc + complex128(u[pos, 1, 0]) * aj1 * flip_val
                sum_waveform[irec, 3, ik] = acc
            elif group == 4:
                acc4 = sum_waveform[irec, 4, ik]
                acc5 = sum_waveform[irec, 5, ik]
                for pos in range(start, end):
                    i = i_list[pos]
                    z = z_table[i, irec]
                    aj0 = aj0_table[i, irec]
                    aj1 = aj1_table[i, irec]
                    nf = (complex128(u[pos, 1, 1]) +
                          complex128(u[pos, 1, 2])) * aj1 / z
                    acc4 = acc4 + (complex128(u[pos, 1, 1]) * aj0 - nf)
                    acc5 = acc5 + (complex128(u[pos, 1, 2]) * aj0 - nf)
                sum_waveform[irec, 4, ik] = acc4
                sum_waveform[irec, 5, ik] = acc5
            elif group == 5:
                acc = sum_waveform[irec, 6, ik]
                for pos in range(start, end):
                    aj2 = aj2_table[i_list[pos], irec]
                    acc = acc + complex128(u[pos, 2, 0]) * aj2 * flip_val
                sum_waveform[irec, 6, ik] = acc
            else:
                acc7 = sum_waveform[irec, 7, ik]
                acc8 = sum_waveform[irec, 8, ik]
                for pos in range(start, end):
                    i = i_list[pos]
                    z = z_table[i, irec]
                    aj1 = aj1_table[i, irec]
                    aj2 = aj2_table[i, irec]
                    nf = 2. * (complex128(u[pos, 2, 1]) +
                               complex128(u[pos, 2, 2])) * aj2 / z
                    acc7 = acc7 + (complex128(u[pos, 2, 1]) * aj1 - nf)
                    acc8 = acc8 + (complex128(u[pos, 2, 2]) * aj1 - nf)
                sum_waveform[irec, 7, ik] = acc7
                sum_waveform[irec, 8, ik] = acc8
