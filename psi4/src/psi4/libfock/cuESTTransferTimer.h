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

#ifndef CUEST_TRANSFER_TIMER_H
#define CUEST_TRANSFER_TIMER_H

#ifdef USING_cuEST

#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <string>

#include "psi4/libqt/qt.h"

namespace psi {
namespace cuest_common {

// Cached value of the CUEST_TIME_TRANSFERS option; refreshed by ensure_cuest_initialized().
extern bool time_transfers;

// Drop-in replacement for cudaMemcpy that charges host<->device copies to the
// "cuEST: H2D" / "cuEST: D2H" timers in timer.dat.  The device is synchronized
// before the copy so queued kernels are not billed to it, and after the copy
// because a pageable H2D cudaMemcpy may return before the DMA has landed.
// With CUEST_TIME_TRANSFERS false this is a plain, unsynchronized cudaMemcpy.
inline cudaError_t timed_memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (!time_transfers) return cudaMemcpy(dst, src, count, kind);

    static const std::string h2d_key = "cuEST: H2D";
    static const std::string d2h_key = "cuEST: D2H";

    const std::string* key = nullptr;
    if (kind == cudaMemcpyHostToDevice) key = &h2d_key;
    if (kind == cudaMemcpyDeviceToHost) key = &d2h_key;
#ifdef _OPENMP
    // Serial psi4 timers must not be touched from inside a parallel region.
    if (omp_in_parallel()) key = nullptr;
#endif
    if (key == nullptr) return cudaMemcpy(dst, src, count, kind);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return err;

    timer_on(*key);
    err = cudaMemcpy(dst, src, count, kind);
    cudaError_t sync_err = cudaDeviceSynchronize();
    timer_off(*key);

    return (err != cudaSuccess) ? err : sync_err;
}

}  // namespace cuest_common
}  // namespace psi

#endif  // USING_cuEST
#endif  // CUEST_TRANSFER_TIMER_H
