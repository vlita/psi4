#ifdef USING_cuEST
#include <cublas_v2.h>
// extern cublasHandle_t cublas_handle;

// Helper macros

// #define CHECK_CUDA(call) \
//     do { \
//         cudaError_t err__ = (call); \
//         if (err__ != cudaSuccess) return static_cast<int>(err__); \
//     } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t st__ = (call); \
        if (st__ != CUBLAS_STATUS_SUCCESS) return static_cast<int>(st__); \
    } while (0)

namespace psi {

void C_DGEMM_CUBLAS(
    cublasHandle_t handle,
    bool transa, bool transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int offa, int lda,
    const double *B, int offb, int ldb,
    const double *beta,
    double *C, int ldc) {
    
    // swap argument order as cublas expects column-major buffers
    CHECK_CUBLAS(cublasDgemm(handle,
                (transb ? CUBLAS_OP_T : CUBLAS_OP_N), (transa ? CUBLAS_OP_T : CUBLAS_OP_N),
                n, m, k,
                alpha,
                (B + offb * sizeof(double)), int ldb,
                (A + offa * sizeof(double)), int lda,
                beta,
                C, ldc));
}

void C_DDOT_CUBLAS(
    cublasHandle_t handle, int n,
    const double *x, int offx, int incx,
    const double *y, int offy, int incy,
    double *result) {

    CHECK_CUBLAS(cublasDdot(handle, n,
                (x + offx * sizeof(double)), incx,
                (y + offy * sizeof(double)), incy,
                result));
}

} // namespace psi
#endif
