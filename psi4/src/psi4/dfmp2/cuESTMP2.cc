#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "psi4/liboptions/liboptions.h"

#include <cuest.h>

using namespace psi;

extern cuestHandle_t cuest_handle;
extern cublasHandle_t cublas_handle;

namespace cuest_dfmp2 {

DFMP2::DFMP2(
    SharedWavefunction ref_wfn, 
    std::shared_ptr<BasisSet> primary, 
    std::shared_ptr<BasisSet> auxiliary, 
    Options& options
    ) : Wavefunction(options) {

    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;
    cuESTDFMP2::common_init();
}

cuESTDFMP2::common_init() {

    int nocc = Cocc_->colspi()[0];
    int nvir = Cvir_->colspi()[0];
    int naux = cuest_auxiliary_basis_->nbf();
    
    int nocc2 = nocc * nocc;
    int nvir2 = nvir * nvir;

    double* Bia_ = nullptr;
    double* iajb_ = nullptr;
    double* ibja_ = nullptr;
    double* pair_amplitudes_ = nullptr;
    
    cuest_common::ensure_cuest_initialized();

    cuest_primary_basis_ = primary->cuest_basis();
    cuest_auxiliary_basis_ = auxiliary->cuest_basis();

    std::shared_ptr<cuESTJK> cuest_jk = std::dynamic_pointer_cast<cuESTJK>(jk_);
    cuestDFIntPlan_t cuest_df_plan_ = cuest_jk->cuest_df_plan();

    variable_buffer_size_bytes_ = 2000000000;
    // cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    // cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

}


cuESTDFMP2::~cuESTDFMP2() {
    // free(persistentWorkspaceDescriptor);
    // free(temporaryWorkspaceDescriptor);
}
  
void cuESTDFMP2::form_Bia() {

    // get MO coefficient matricies from wfn
    auto Cocc = Ca_subset("AO", "ACTIVE_OCC");
    auto Cvir = Ca_subset("AO", "ACTIVE_VIR");

    // pointers to row-major data
    double* Cleft = Cocc->pointer()[0];
    double* Cright = Cvir->pointer()[0];

    // set up device buffers
    double* d_Cleft = nullptr;
    double* d_Cleft = nullptr;

    CHECK_CUDA(cudaMalloc((void**) &d_Cleft, sizeof(double) * nocc * nao));
    CHECK_CUDA(cudaMalloc((void**) &d_Cright, sizeof(double) * nvir * nao));

    // copy data to device
    cudaMemcpy(d_Cleft, Cleft, sizeof(double) * nocc * nao, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cleft, Cleft, sizeof(double) * nvir * nao, cudaMemcpyHostToDevice);

    // code below is adapeted from the CUDALibrarySamples MO ints example

    // presently there are no configurable parameters for MO ints
    cuestDFMOIntegralsComputeParameters_t params;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        &params));

    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = variable_buffer_size_bytes_;

    // workspace query to determine temporary space needed to form Bia
    CHECK_CUEST(cuestDFMOIntegralsComputeWorkspaceQuery(
        cuest_handle,
        cuest_df_plan_,
        params,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        1,
        nocc,
        nvir,
        d_Cleft,
        d_Cright,
        Bia_));

    cuestWorkspace_t* temporaryDFMOWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    // compute DF MO integrals Bia = (A|ia)
    CHECK_CUEST(cuestDFMOIntegralsCompute(
        cuest_handle,
        cuest_df_plan_,
        params,
        variableBufferSize,
        temporaryDFMOWorkspace,
        1,
        nocc,
        nvir,
        d_Cleft,
        d_Cright,
        Bia_));

    // free buffers
    free(variableBufferSize);
    cuest_common::freeWorkspace(temporaryDFMOWorkspace);

    CHECK_CUEST(cuestParametersDestroy(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        params));

    cudaFree((void*) d_Cleft);
    cudaFree((void*) d_Cright);
}


void cuESTDFMP2::compute_energy_packed() {

    // set up device buffers
    double* d_os = nullptr;
    double* d_ss = nullptr;
    
    CHECK_CUDA(cudaMalloc((void**) &d_os, sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &d_ss, sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &Bia_, naux * nocc * nvir * sizeof(double)));

    // size buffers based on available memory
    int slice_count = cuESTDFMP2::slice_count();
    size_t size_in_bytes = sizeof(double) * nvir2 * nocc * slice_count;
    
    CHECK_CUDA(cudaMalloc((void**) &iajb_, size_in_bytes));
    CHECK_CUDA(cudaMalloc((void**) &ibja_, size_in_bytes));
    CHECK_CUDA(cudaMalloc((void**) &pair_amplitudes_, size_in_bytes));

    // final energies 
    double mp2_os_energy;
    double mp2_ss_energy;
    
    for (int i = 0; i < (nocc + slice_count - 1) / slice_count; i++) {
        int iter_slices = min(slice_count, nocc - i * slice_count);

        // ==> (i a | j b)[ia_1 : ia_2, jb] = sum_A B[A, ia_1 : ia_2] * B[A, jb] <==
        C_DGEMM_CUBLAS(
            transa=True,
            transb=False,
            m=iter_slices*nvir,
            n=nocc*nvir,
            k=naux,
            alpha=1.0,
            a=Bia_,
            offa=i*slice_count*nvir,
            lda=nocc*nvir,
            b=Bia_,
            offb=0,
            ldb=nocc*nvir,
            beta=0.0,
            c=iajb_,
            offc=0,
            ldc=nocc*nvir
            );

        double mp2_os_energy;
        double mp2_ss_energy;
        
    }
}

void cuESTDFMP2::compute_energy_low_mem() {
    
}


int cuESTDFMP2::slice_count() {
    size_t free, total;
    cudaMemGetInfo(size_t *free, size_t *total);
    
    size_t slices_possible = static_cast<size_t>((static_cast<size_t>(free * 0.8) / 3) / (sizeof(double) * nvir2_ * nocc_);
    int slice_count = min(slices_possible, nocc);

    if(slice_count == 0) {
        throw PSIEXCEPTION("Insufficient memory for packed GEMM algorithm")
    }

    return slice_count;
}



} // namespace cuestDFMP2