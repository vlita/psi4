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

cuest_common::ensure_cuest_initialized();
variable_buffer_size_bytes_ = 2000000000

cuest_primary_basis_ = primary->cuest_basis();
cuest_auxiliary_basis_ = auxiliary->cuest_basis();

std::shared_ptr<cuESTJK> cuest_jk = std::dynamic_pointer_cast<cuESTJK>(jk_);
cuestDFIntPlan_t cuest_df_plan_ = cuest_jk->cuest_df_plan();

cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

// DESTRUCTOR
//free temporary and persistent workspaces

void cuESTDFMP2::Bia(
) {
    // input: cuest scf metadata, device buffer for Bia
    // output: Bia

    // needs the following either explicitly or stored on the cuest_df_plan object:
    // 1. cuest_primary basis
    // 2. cuest_auxillary basis
    // 3. cuest_ao pair list
    // 4. df_fitting_eigenvalue_cutoff

    // needs the following explicitly:
    // 1. cuest_handle
    // 2. cuest_df_plan
    // 3. MO integrals compute parameters (no configurable parameters at this time)
    // 4. variableBufferSize
    // 5. temporary workspace
    // 6. numCoefficientMatrices (1, determined by the fact that we are using an rhf reference I think)
    // 7. numLeftOrbitals (number of occupied orbitals, nocc) 
    // 8. numRightOrbitals (number of virtual orbitals, nvirt)
    // 9. leftCoefficientMatrices [nocc, nao] 
    // 10. rightCoefficientMatrices [nvir, nao] both of these quantities we can pull off of Ca.
    // 11. outTensors (Bia buffer)    

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
        
    cuestDFMOIntegralsComputeParameters_t params;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        &params));

    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = variable_buffer_size_bytes_;

    // workspace query to determine temporary space needed to form Bia
    checkCuestErrors(cuestDFMOIntegralsComputeWorkspaceQuery(
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
        Bia));

    cuestWorkspace_t* temporaryDFMOWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    // compute DF MO integrals Bia = (A|ia)
    checkCuestErrors(cuestDFMOIntegralsCompute(
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
        Bia));

    free(variableBufferSize);
    cuest_common::freeWorkspace(temporaryDFMOWorkspace);

    checkCuestErrors(cuestParametersDestroy(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        params));

    // free buffers
    cudaFree((void*) d_Cleft);
    cudaFree((void*) d_Cright);

}


void cuESTDFMP2::compute_energy () {
    // double* d_Bia = nullptr;
    // cudaMalloc((void**) &d_Bia, naux * nocc * nvir * sizeof(double);

}





} // namespace cuestDFMP2