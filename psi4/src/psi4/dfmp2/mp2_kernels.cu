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

// blockIdx.z loops over (i, j) blocks, since npairs can exceed the 65535 grid z limit
static const int max_grid_z = 65535;

// this kernel is adapted from the CUDALibrarySamples mp2 code
__global__
void make_pair_amplitudes_sliced(
    long long i_start,
    long long nocc,
    long long nvir,
    long long npairs,
    const double* iajb,
    const double* eps_occ,
    const double* eps_vir,
    double* iajb_d
) {
    // for maximum memory coalescing, idx x handles b and idx y handles a
    long long rel_a = (long long) blockIdx.y * blockDim.y + threadIdx.y;
    long long rel_b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (rel_a >= nvir || rel_b >= nvir) return;

    long long ncols = nocc * nvir;
    for (long long pair = blockIdx.z; pair < npairs; pair += gridDim.z) {
        long long i = i_start + pair / nocc;
        long long j = pair % nocc;
        long long idx = (pair / nocc) * nvir * ncols + j * nvir + rel_a * ncols + rel_b;

        double denominator = eps_occ[i] + eps_occ[j] - eps_vir[rel_a] - eps_vir[rel_b];
        iajb_d[idx] = iajb[idx] * (fabs(denominator) < 1e-20 ? 1e20 : 1.0 / denominator);
    }
}

// this kernel is adapted from the CUDALibrarySamples mp2 code
__global__
void make_sliced_transpose(
    long long nocc,
    long long nvir,
    long long npairs,
    const double* iajb,
    double* ibja
) {
    long long rel_a = (long long) blockIdx.y * blockDim.y + threadIdx.y;
    long long rel_b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (rel_a >= nvir || rel_b >= nvir) return;

    long long ncols = nocc * nvir;
    for (long long pair = blockIdx.z; pair < npairs; pair += gridDim.z) {
        long long block_start = (pair / nocc) * nvir * ncols + (pair % nocc) * nvir;
        ibja[block_start + rel_b * ncols + rel_a] = iajb[block_start + rel_a * ncols + rel_b];
    }
}

void pair_amplitudes_sliced(
    int i_start,
    int nocc,
    int nvir,
    int npairs,
    const double* iajb,
    const double* eps_occ,
    const double* eps_vir,
    double* iajb_d,
    cudaStream_t stream
) {
    // schedule thread blocks of 256 elements each (16x16)
    dim3 DimGrid((nvir + 16 - 1) / 16, (nvir + 16 - 1) / 16, npairs < max_grid_z ? npairs : max_grid_z);
    dim3 DimBlock(16, 16, 1);
    make_pair_amplitudes_sliced<<<DimGrid, DimBlock, 0, stream>>>(i_start, nocc, nvir, npairs, iajb, eps_occ, eps_vir, iajb_d);
}

void sliced_transpose(
    int nocc,
    int nvir,
    int npairs,
    const double* iajb,
    double* ibja,
    cudaStream_t stream
) {
    // schedule thread blocks of 256 elements each (16x16)
    dim3 DimGrid((nvir + 16 - 1) / 16, (nvir + 16 - 1) / 16, npairs < max_grid_z ? npairs : max_grid_z);
    dim3 DimBlock(16, 16, 1);
    make_sliced_transpose<<<DimGrid, DimBlock, 0, stream>>>(nocc, nvir, npairs, iajb, ibja);
}

#endif // USING_cuEST