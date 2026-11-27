#ifdef USING_cuEST

#include "mp2_kernels.h"

// this kernel is adapted from the CUDALibrarySamples mp2 code
__global__
void make_pair_amplitudes_ij(
    long long i,
    long long j,
    long long nvir,
    const double* iajb,
    const double* eps_occ,
    const double* eps_vir,
    double* iajb_d
) {
    // for maximum memory coalescing, idx x handles b and idx y handles a
    long long rel_a = (long long) blockIdx.y * blockDim.y + threadIdx.y;
    long long rel_b = (long long) blockIdx.x * blockDim.x + threadIdx.x;

    if (rel_a < nvir && rel_b < nvir) {
        double denominator = eps_occ[i] + eps_occ[j] - eps_vir[rel_a] - eps_vir[rel_b];
        double numerator = iajb[rel_a * nvir + rel_b];
        iajb_d[rel_a * nvir + rel_b] = numerator * (fabs(denominator) < 1e-20 ? 1e20 : 1.0 / denominator);
    }
}

void pair_amplitudes_ij(
    int i,
    int j,
    int nvir,
    const double* iajb,
    const double* eps_occ,
    const double* eps_vir,
    double* iajb_d,
    cudaStream_t stream
) {
    // schedule thread blocks of 256 elements each (16x16)
    dim3 DimGrid((nvir + 16 - 1) / 16, (nvir + 16 - 1) / 16, 1);
    dim3 DimBlock(16, 16, 1);
    make_pair_amplitudes_ij<<<DimGrid, DimBlock, 0, stream>>>(i, j, nvir, iajb, eps_occ, eps_vir, iajb_d);
}

// for maximum memory coalescing in the kernels below, idx x runs along a row (columns) and idx y runs down the rows

__global__
void make_pair_tau(
    long long nvir,
    const double* t,
    double* T
) {
    long long a = (long long) blockIdx.y * blockDim.y + threadIdx.y;
    long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;

    if (a < nvir && b < nvir) {
        T[a * nvir + b] = 2.0 * t[a * nvir + b] - t[b * nvir + a];
    }
}

void pair_tau(
    int nvir,
    const double* t,
    double* T,
    cudaStream_t stream
) {
    dim3 DimGrid((nvir + 16 - 1) / 16, (nvir + 16 - 1) / 16, 1);
    dim3 DimBlock(16, 16, 1);
    make_pair_tau<<<DimGrid, DimBlock, 0, stream>>>(nvir, t, T);
}

__global__
void make_apply_denominator(
    long long nvir,
    long long nocc,
    bool invert,
    double alpha,
    double beta,
    const double* eps_occ,
    const double* eps_vir,
    const double* x,
    double* y
) {
    long long a = (long long) blockIdx.y * blockDim.y + threadIdx.y;
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;

    if (a < nvir && i < nocc) {
        long long idx = a * nocc + i;
        double denominator = eps_vir[a] - eps_occ[i];
        double weight = invert ? 1.0 / denominator : denominator;
        y[idx] = beta * y[idx] + alpha * x[idx] * weight;
    }
}

void apply_denominator(
    int nvir,
    int nocc,
    bool invert,
    double alpha,
    double beta,
    const double* eps_occ,
    const double* eps_vir,
    const double* x,
    double* y,
    cudaStream_t stream
) {
    dim3 DimGrid((nocc + 16 - 1) / 16, (nvir + 16 - 1) / 16, 1);
    dim3 DimBlock(16, 16, 1);
    make_apply_denominator<<<DimGrid, DimBlock, 0, stream>>>(nvir, nocc, invert, alpha, beta, eps_occ, eps_vir, x, y);
}

__global__
void make_scale_rows(
    long long nrows,
    long long ncols,
    const double* scales,
    double* matrix
) {
    long long row = (long long) blockIdx.y * blockDim.y + threadIdx.y;
    long long col = (long long) blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nrows && col < ncols) {
        matrix[row * ncols + col] *= scales[row];
    }
}

void scale_rows(
    int nrows,
    int ncols,
    const double* scales,
    double* matrix,
    cudaStream_t stream
) {
    dim3 DimGrid((ncols + 16 - 1) / 16, (nrows + 16 - 1) / 16, 1);
    dim3 DimBlock(16, 16, 1);
    make_scale_rows<<<DimGrid, DimBlock, 0, stream>>>(nrows, ncols, scales, matrix);
}

__global__
void make_scatter_block(
    long long nrows,
    long long ncols,
    long long ld_dst,
    double alpha,
    const double* src,
    double* dst
) {
    long long row = (long long) blockIdx.y * blockDim.y + threadIdx.y;
    long long col = (long long) blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nrows && col < ncols) {
        dst[row * ld_dst + col] = alpha * src[row * ncols + col];
    }
}

void scatter_block(
    int nrows,
    int ncols,
    int ld_dst,
    double alpha,
    const double* src,
    double* dst,
    cudaStream_t stream
) {
    dim3 DimGrid((ncols + 16 - 1) / 16, (nrows + 16 - 1) / 16, 1);
    dim3 DimBlock(16, 16, 1);
    make_scatter_block<<<DimGrid, DimBlock, 0, stream>>>(nrows, ncols, ld_dst, alpha, src, dst);
}

__global__
void make_set_diagonal(
    long long count,
    long long ld,
    double value,
    double* matrix
) {
    long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        matrix[idx * ld + idx] = value;
    }
}

void set_diagonal(
    int count,
    int ld,
    double value,
    double* matrix,
    cudaStream_t stream
) {
    make_set_diagonal<<<(count + 256 - 1) / 256, 256, 0, stream>>>(count, ld, value, matrix);
}

__global__
void make_energy_weighted_P(
    long long nmo,
    long long nocc,
    const double* eps,
    const double* P,
    double* W
) {
    long long p = (long long) blockIdx.y * blockDim.y + threadIdx.y;
    long long q = (long long) blockIdx.x * blockDim.x + threadIdx.x;

    if (p < nmo && q < nmo) {
        double weight;
        if (p >= nocc && q < nocc) {
            weight = eps[q];
        } else if (p < nocc && q >= nocc) {
            weight = eps[p];
        } else {
            weight = 0.5 * (eps[p] + eps[q]);
        }
        W[p * nmo + q] = weight * P[p * nmo + q];
    }
}

void make_W(
    int nmo,
    int nocc,
    const double* eps,
    const double* P,
    double* W,
    cudaStream_t stream
) {
    dim3 DimGrid((nmo + 16 - 1) / 16, (nmo + 16 - 1) / 16, 1);
    dim3 DimBlock(16, 16, 1);
    make_energy_weighted_P<<<DimGrid, DimBlock, 0, stream>>>(nmo, nocc, eps, P, W);
}

#endif // USING_cuEST