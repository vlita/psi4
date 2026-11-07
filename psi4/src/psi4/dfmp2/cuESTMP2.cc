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

// constructor should give us a way to check the status of cuest, 
// access the primary and auxillary basis sets, as well as any other 
// data we need to pull from converged HF

// CONSTRUCTOR

nocc = 
nvir = 
nao = 

double* Bia_ = nullptr;
cudaMalloc((void**) &Bia_, naux * nocc * nvir * sizeof(double);

cuest_common::ensure_cuest_initialized();
variable_buffer_size_bytes_ = 2000000000

cuest_primary_basis_ = primary->cuest_basis();MN
cuest_auxiliary_basis_ = auxiliary->cuest_basis();

std::shared_ptr<cuESTJK> cuest_jk = std::dynamic_pointer_cast<cuESTJK>(jk_);
cuestDFIntPlan_t cuest_df_plan_ = cuest_jk->cuest_df_plan();

cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

// DESTRUCTOR
//free temporary and persistent workspaces

void cuESTDFMP2::form_Bia() {

    // get MO coefficient matricies from wfn
    auto Cocc = wfn->Ca_subset("AO", "ACTIVE_OCC");
    auto Cvir = wfn->Ca_subset("AO", "ACTIVE_VIR");

    // pointers to row-major data
    double* Cleft = Cocc->pointer()[0];
    double* Cright = Cvir->pointer()[0];

    // set up device buffers
    double* d_Cleft = nullptr;
    double* d_Cleft = nullptr;

    cudaMalloc((void**) &d_Cleft, sizeof(double) * nocc * nao);
    cudaMalloc((void**) &d_Cright, sizeof(double) * nvir * nao);

    // copy data to device
    cudaMemcpy(d_Cleft, Cleft, sizeof(double) * nocc * nao, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cleft, Cleft, sizeof(double) * nvir * nao, cudaMemcpyHostToDevice);

    // code below is adapeted from the CUDALibrarySamples df_mo_integrals example

    // presently there are no configurable parameters for this computation
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


void cuESTDFMP2::compute_energy () {

    // do we have enough room to pack (ia|jb) = (A|ia)(A|jb) into a few large DGEMMs?
    size_t free, total;
    cudaMemGetInfo(size_t *free, size_t *total);
    

}





} // namespace cuestDFMP2