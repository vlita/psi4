/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#ifndef THREE_INDEX_CUESTMOINTS_H
#define THREE_INDEX_CUESTMOINTS_H

#ifdef USING_cuEST

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuest.h>

#include "psi4/libmints/typedefs.h"

namespace psi {

class BasisSet;
class Options;

/**
 * GPU density-fitted MO integrals via cuEST.
 *
 * Owns the cuEST AO pair list and DF integral plan built from the given
 * (primary, auxiliary) pair. The auxiliary basis decides what the integrals
 * are, so pass the basis for the method (e.g. DF_BASIS_MP2), not the SCF's.
 *
 * The cuestAOBasis_t handles themselves are owned by the BasisSet objects,
 * which this class keeps alive.
 */
class cuESTMOInts {
   protected:
    std::shared_ptr<BasisSet> primary_;
    std::shared_ptr<BasisSet> auxiliary_;

    double pq_threshold_;
    int32_t integral_direct_;

    cuestAOBasis_t cuest_primary_basis_ = nullptr;
    cuestAOBasis_t cuest_auxiliary_basis_ = nullptr;
    cuestAOPairList_t cuest_pair_list_ = nullptr;
    cuestDFIntPlan_t cuest_df_plan_ = nullptr;

    cuestWorkspace_t* cuest_pair_list_ws_ptr_ = nullptr;
    cuestWorkspace_t* cuest_dfint_plan_ws_ptr_ = nullptr;

    // One-electron plan on the same pair list, only built if a one-electron derivative is requested
    cuestOEIntPlan_t cuest_oe_plan_ = nullptr;
    cuestWorkspace_t* cuest_oeint_plan_ws_ptr_ = nullptr;

    void common_init();
    void build_oe_plan();

   public:
    cuESTMOInts(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);
    ~cuESTMOInts();

    cuESTMOInts(const cuESTMOInts&) = delete;
    cuESTMOInts& operator=(const cuESTMOInts&) = delete;

    /**
     * Fitted 3-index MO integrals B(Q|lr) = sum_mn B(Q|mn) C_ml C_nr for each (left, right) pair.
     *
     * nleft / nright:    number of left / right orbitals of each pair.
     * d_Cleft / d_Cright: DEVICE buffers of the concatenated coefficient matrices, each [nmo, nbf].
     * d_out:             caller-owned DEVICE buffer of naux * sum_i nleft[i] * nright[i] doubles,
     *                    one row-major [Q][l][r] tensor per pair, concatenated.
     */
    void compute_3c_ints(const std::vector<uint64_t>& nleft, const std::vector<uint64_t>& nright, const double* d_Cleft,
                         const double* d_Cright, double* d_out);

    /**
     * Nuclear gradient of sum_A sum_lr gamma3(A|lr) B(A|lr), the counterpart of compute_3c_ints.
     *
     * nleft / nright / d_Cleft / d_Cright: as in compute_3c_ints.
     * d_tpdm:   DEVICE three-center TPDM, laid out exactly like the d_out of compute_3c_ints. It is
     *           assumed bra-ket symmetric; any pair symmetry or spin factor must already be folded in.
     * d_grad:   DEVICE buffer of natom * 3 doubles, overwritten (not accumulated).
     */
    void compute_3c_derivative(const std::vector<uint64_t>& nleft, const std::vector<uint64_t>& nright,
                               const double* d_Cleft, const double* d_Cright, const double* d_tpdm, double* d_grad);

    /// J(D) for a DEVICE symmetric AO density [nbf, nbf]; d_J is overwritten.
    void compute_J(const double* d_D, double* d_J);

    /**
     * K_mn = sum_ls (ml|ns) sum_i L_il R_is with DEVICE coefficient buffers d_Cleft, d_Cright of
     * shape [nvec, nbf]; d_K [nbf, nbf] is overwritten.
     */
    void compute_K_nonsymmetric(uint64_t nvec, const double* d_Cleft, const double* d_Cright, double* d_K);

    /// One-electron derivative integrals contracted with a DEVICE AO matrix [nbf, nbf]; d_grad [natom, 3] is overwritten.
    void compute_overlap_derivative(const double* d_W, double* d_grad);
    void compute_kinetic_derivative(const double* d_D, double* d_grad);
    /// Point charges q at xyz (DEVICE, [ncharge] and [ncharge, 3]); basis and charge parts come back separately.
    void compute_potential_derivative(uint64_t ncharge, const double* d_xyz, const double* d_q, const double* d_D,
                                      double* d_grad_basis, double* d_grad_charge);
};

}  // namespace psi

#endif  // USING_cuEST
#endif  // THREE_INDEX_CUESTMOINTS_H
