#pragma once
#ifdef USING_cuEST

#include <cuda_runtime.h>

// GPU kernel to update iajb with energy denominator eps[i] + eps[j] - eps[a] - eps[b] for fixed i,j
void pair_amplitudes_ij(int i, int j, int nvir, const double* iajb, const double* eps_occ, const double* eps_vir, double* iajb_d, cudaStream_t stream);

// High-memory variants. The slice buffers hold (i a | j b) for i in [i_start, i_start + nslice) as a
// (nslice * nvir) x (nocc * nvir) row-major matrix; npairs = nslice * nocc is the number of (i, j) blocks.

// GPU kernel to apply the energy denominator to every (i, j) block of a slice
void pair_amplitudes_sliced(int i_start, int nocc, int nvir, int npairs, const double* iajb, const double* eps_occ, const double* eps_vir, double* iajb_d, cudaStream_t stream);

// GPU kernel to form (i b | j a) from (i a | j b) within every (i, j) block of a slice
void sliced_transpose(int nocc, int nvir, int npairs, const double* iajb, double* ibja, cudaStream_t stream);

#endif // USING_cuEST