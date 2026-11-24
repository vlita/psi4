#pragma once
#ifdef USING_cuEST

#include <cuda_runtime.h>

// GPU kernel to update iajb with energy denominator eps[i] + eps[j] - eps[a] - eps[b] for fixed i,j
void pair_amplitudes_ij(int i, int j, int nvir, const double* iajb, const double* eps_occ, const double* eps_vir, double* iajb_d, cudaStream_t stream);

// The kernels below are adapted from the CUDALibrarySamples mp2-grad-1 example; all matrices are row-major

// T[a, b] = 2 t[a, b] - t[b, a] for one (nvir, nvir) amplitude pair block (t and T must not alias)
void pair_tau(int nvir, const double* t, double* T, cudaStream_t stream);

// y[a, i] = beta y[a, i] + alpha x[a, i] * w[a, i], where w = (eps_vir[a] - eps_occ[i]) or its inverse (x and y may alias)
void apply_denominator(int nvir, int nocc, bool invert, double alpha, double beta, const double* eps_occ, const double* eps_vir, const double* x, double* y, cudaStream_t stream);

// matrix[r, c] *= scales[r]
void scale_rows(int nrows, int ncols, const double* scales, double* matrix, cudaStream_t stream);

// dst[r * ld_dst + c] = alpha src[r * ncols + c]; dst points at the corner of the target block of a wider matrix
void scatter_block(int nrows, int ncols, int ld_dst, double alpha, const double* src, double* dst, cudaStream_t stream);

// matrix[d * ld + d] = value for d < count
void set_diagonal(int count, int ld, double value, double* matrix, cudaStream_t stream);

// W[p, q] = w[p, q] P[p, q], with w = eps of the occupied index in the occ-vir blocks and the average of the two elsewhere
void make_W(int nmo, int nocc, const double* eps, const double* P, double* W, cudaStream_t stream);

#endif // USING_cuEST