#include "cuESTMOInts.h"

#ifdef USING_cuEST

#include <cuest.h>
#include "psi4/libfock/cuESTCommon.h"
#include "psi4/libmints/basisset.h"

namespace psi {

cuESTMOInts::cuESTMOInts(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, SharedMatrix C, int nocc, int nvir)
    : primary_(primary),
      auxiliary_(auxiliary)) {
        
    common_init();
}

cuESTMOInts::~cuESTMOInts() {
    if (cuest_df_plan_) {
        CHECK_CUEST(cuestDFIntPlanDestroy(cuest_df_plan_));
        cuest_df_plan_ = nullptr;
    }
    if (cuest_pair_list_) {
        CHECK_CUEST(cuestAOPairListDestroy(cuest_pair_list_));
        cuest_pair_list_ = nullptr;
    }
    if (cuest_pair_list_ws_ptr_) {
        cuest_common::freeWorkspace(cuest_pair_list_ws_ptr_);
        cuest_pair_list_ws_ptr_ = nullptr;
    }
    if (cuest_dfint_plan_ws_ptr_) {
        cuest_common::freeWorkspace(cuest_dfint_plan_ws_ptr_);
        cuest_dfint_plan_ws_ptr_ = nullptr;
    }
}

void cuESTMOInts::common_init() {

    cuest_ri_basis = ribasis_->cuest_basis();
    cuest_primary_basis = primary_->cuest_basis();

    condition_ = options.get_double("DF_FITTING_CONDITION");
    pq_threshold_ = options.get_double("INTS_TOLERANCE");

    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    // AO Pair List
    auto mol = primary_->molecule();
    size_t natom = mol->natom();
    
    cuestAOPairListParameters_t pair_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_AOPAIRLIST_PARAMETERS, reinterpret_cast<void**>(&pair_params)));

    CHECK_CUEST(cuestAOPairListCreateWorkspaceQuery(cuest_handle, cuest_primary_basis_,
        static_cast<uint64_t>(natom), mol->geometry().pointer()[0], pq_threshold_, pair_params, persistentWorkspaceDescriptor, temporaryWorkspaceDescriptor, nullptr));

    cuest_pair_list_ws_ptr_ = cuest_common::allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryPairListWorkspace = cuest_common::allocateWorkspace(temporaryWorkspaceDescriptor);

    CHECK_CUEST(cuestAOPairListCreate(cuest_handle, cuest_primary_basis_,
        static_cast<uint64_t>(natom), mol->geometry().pointer()[0], pq_threshold_, pair_params, cuest_pair_list_ws_ptr_, temporaryPairListWorkspace, &cuest_pair_list_));

    cuest_common::freeWorkspace(temporaryPairListWorkspace);
    CHECK_CUEST(cuestParametersDestroy(CUEST_AOPAIRLIST_PARAMETERS, pair_params));

    // DF Int Plan
    cuestDFIntPlanParameters_t dfint_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_DFINTPLAN_PARAMETERS, reinterpret_cast<void**>(&dfint_params)));

    // this option will only work for the MO Ints transform
    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFINTPLAN_PARAMETERS, 
        dfint_params,
        CUEST_DFINTPLAN_PARAMETERS_THREE_INDEX_INTEGRAL_DIRECT,
        &static_cast<uint64_t>(integral_direct_),
        sizeof(uint64_t)));

    CHECK_CUEST(cuestDFIntPlanCreateWorkspaceQuery(
        cuest_handle,
        cuest_primary_basis_, 
        cuest_auxiliary_basis_, 
        cuest_pair_list_,
        dfint_params, 
        persistentWorkspaceDescriptor, 
        temporaryWorkspaceDescriptor, 
        nullptr));

    cuest_dfint_plan_ws_ptr_ = cuest_common::allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryDFIntPlanWorkspace = cuest_common::allocateWorkspace(temporaryWorkspaceDescriptor);;

    CHECK_CUEST(cuestDFIntPlanCreate(cuest_handle,
        cuest_primary_basis_, cuest_auxiliary_basis_, cuest_pair_list_,
        dfint_params, cuest_dfint_plan_ws_ptr_, temporaryDFIntPlanWorkspace, &cuest_df_plan_));

    cuest_common::freeWorkspace(temporaryDFIntPlanWorkspace);
    CHECK_CUEST(cuestParametersDestroy(CUEST_DFINTPLAN_PARAMETERS, dfint_params));

    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);
}

void cuESTMOInts::compute_3c_ints(){
    
}

#endif // using cuEST