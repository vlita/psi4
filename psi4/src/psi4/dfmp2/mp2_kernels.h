#ifdef USING_cuEST

#include <cuda_runtime.h>

// GPU kernel to update iajb with energy denominator eps[i] + eps[j] - eps[a] - eps[b] for fixed i,j
void pair_amplitudes_ij(int i, int j, int nvir, const double* eps, const double* iajb, double* iajb_d, cudaStream_t stream);

#endif // USING_cuEST