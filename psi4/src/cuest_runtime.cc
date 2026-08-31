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
// The interface to cuEST was contributed by NVIDIA under the following terms:
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LGPL-3.0-only

/*! \file cuest_runtime.cc
 *  \brief GPU discovery, health checking, and cuEST/CUDA handle lifetime.
 *
 *  Split out of core.cc, which is the pybind11 module definition and has no
 *  business carrying several hundred lines of CUDA diagnostics.  CMake only
 *  compiles this file when cuEST is enabled, but it is additionally guarded so
 *  that it degrades to an empty translation unit rather than a wall of errors
 *  if it is ever compiled without.
 *
 *  Initialization is *lazy*: nothing here runs at import time, so a build with
 *  cuEST enabled still imports and runs normally on a machine with no GPU (or
 *  with an unusable one).  The GPU is not touched until the first computation
 *  that actually asks for cuEST calls ensure_cuest_initialized(), and a failure
 *  at that point is a normal, catchable PsiException.
 */

#ifdef USING_cuEST

#include <sstream>
#include <string>
#include <mutex>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuest.h>

#include "psi4/psi4-dec.h"
#include "psi4/libfock/cuESTCommon.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/process.h"

#include "cuest_runtime.h"

using namespace psi;

// Process-global handles. Deliberately at global scope, not in a namespace:
// they are declared `extern` at global scope by every consumer -- libfock
// (cuESTCommon.h, cuESTJK.cc, cubature.cc, v.cc), libmints (basisset.cc,
// mintshelper.cc), libciomr (dsyev_ascending.cc), libscf_solver (hf.cc),
// libpsipcm (cuestpcm.cc), and scfgrad (jk_grad.cc).
//
// A zero handle means "not initialized", and consumers rely on that: the
// cuSOLVER diagonalization in dsyev_ascending.cc falls back to CPU LAPACK on
// cusolver_handle == 0 rather than failing, which is what keeps non-cuEST work
// running in a cuEST-enabled build on a machine with no GPU.
cusolverDnHandle_t cusolver_handle = 0;
cublasHandle_t cublas_handle = 0;
cuestHandle_t cuest_handle = 0;
cudaStream_t stream_handle = 0;

namespace {

cusolverDnHandle_t vxc_cusolver_handle = 0;
cublasHandle_t vxc_cublas_handle = 0;
cuestHandle_t vxc_cuest_handle = 0;
cudaStream_t vxc_stream_handle = 0;

std::mutex initialization_mutex;
bool runtime_configured = false;
int configured_num_gpus = 1;
int default_device_id = -1;
int vxc_device_id = -1;

// Tear down whatever is up, in reverse order of creation, swallowing errors.
// Used both for the ordinary shutdown path and to unwind a partially completed
// initialization.
void cuest_cleanup_noexcept(int device_id, cuestHandle_t& cuest, cusolverDnHandle_t& cusolver,
                            cublasHandle_t& cublas, cudaStream_t& stream) {
    int previous_device = -1;
    cudaGetDevice(&previous_device);
    if (device_id >= 0) cudaSetDevice(device_id);

    if (cuest != 0) {
        cuestDestroy(cuest);
        cuest = 0;
    }
    if (cusolver != 0) {
        cusolverDnDestroy(cusolver);
        cusolver = 0;
    }
    if (cublas != 0) {
        cublasDestroy(cublas);
        cublas = 0;
    }
    if (stream != 0) {
        cudaStreamDestroy(stream);
        stream = 0;
    }

    if (previous_device >= 0 && previous_device != device_id) cudaSetDevice(previous_device);
}

// Some CUDA status codes mean "this GPU or this node is broken/misprovisioned",
// not "Psi4 or cuEST did something wrong". Reported as a bare "Fatal Error"
// they send users hunting through inputs, keywords, and build flags for a
// defect that does not exist -- an uncorrectable ECC fault, for instance, is
// latched by the driver and fails every subsequent context/stream/allocation on
// that device until it is physically reset, so it can even be inherited from an
// unrelated job that ran earlier on the same card. For those codes, name the
// culprit and give the exact command that confirms it.
//
// Returns an empty string for codes that are not known environmental faults.
std::string cuda_unhealthy_environment_hint(cudaError_t err) {
    switch (err) {
        case cudaErrorECCUncorrectable:
            return
                "  The GPU reported an uncorrectable ECC (double-bit) memory error -- a hardware fault in\n"
                "  the device's own memory. The driver latches this state and then fails every context,\n"
                "  stream, and allocation on that GPU until it is physically reset, so the fault may well\n"
                "  have been triggered by an earlier, unrelated job on the same card.\n"
                "\n"
                "  Confirm with:\n"
                "      nvidia-smi -q -d ECC,ROW_REMAPPER\n"
                "      dmesg | grep -i xid          # look for Xid 48, 63, 92, 94, 95, 171\n"
                "  A non-zero 'DRAM Uncorrectable' count, or 'Remapped Rows ... Pending: Yes', confirms it.\n"
                "\n"
                "  Recovery needs 'nvidia-smi -r' (root, with the GPU idle) or a node reboot. Under a batch\n"
                "  scheduler the faulty device often keeps being handed out: ask an administrator to reset\n"
                "  or drain the node, and resubmit onto different hardware.";
        case cudaErrorDevicesUnavailable:
            return
                "  Every visible GPU is busy or is refusing new contexts. Common causes: the device is in\n"
                "  'Exclusive_Process' compute mode and already held by another process, MIG is enabled but\n"
                "  no instance was assigned, or the scheduler granted a GPU that another job still occupies.\n"
                "\n"
                "  Confirm with:\n"
                "      nvidia-smi --query-gpu=index,compute_mode,mig.mode.current --format=csv\n"
                "      nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv\n"
                "  Also check that CUDA_VISIBLE_DEVICES matches what the scheduler actually granted.";
        case cudaErrorSystemNotReady:
            return
                "  The CUDA system stack has not finished initializing. Typically the NVIDIA fabric manager\n"
                "  is not running, or has not converged, on an NVLink/NVSwitch system.\n"
                "\n"
                "  Confirm with:\n"
                "      systemctl status nvidia-fabricmanager\n"
                "      nvidia-smi -q | grep -i -A2 Fabric\n"
                "  This is a node provisioning problem for an administrator to resolve.";
        case cudaErrorInsufficientDriver:
        case cudaErrorSystemDriverMismatch:
            return
                "  The installed NVIDIA kernel driver is older than, or mismatched with, the CUDA runtime\n"
                "  that Psi4/cuEST was built against. This usually means the compute node's driver differs\n"
                "  from the build node's, or a driver upgrade landed without a reboot.\n"
                "\n"
                "  Confirm with:\n"
                "      nvidia-smi --query-gpu=driver_version --format=csv\n"
                "      cat /proc/driver/nvidia/version\n"
                "  Rebuild against the deployed CUDA version, or ask an administrator to align the driver.";
        case cudaErrorNoDevice:
            return
                "  The driver reports no CUDA-capable device at all. Either no GPU was allocated to this\n"
                "  job, CUDA_VISIBLE_DEVICES is empty or names a nonexistent index, or the NVIDIA kernel\n"
                "  module is not loaded on this node.\n"
                "\n"
                "  Confirm with:\n"
                "      echo \"$CUDA_VISIBLE_DEVICES\"; nvidia-smi -L; lsmod | grep nvidia\n"
                "  Under a batch scheduler, verify the job actually requested a GPU (e.g. Slurm --gres=gpu:1).";
        default:
            return std::string();
    }
}

// Compose the text for a failed CUDA call during cuEST startup. `what` is the
// ordinary human-readable description; when the status code is a known
// environmental fault, the health diagnosis is appended. Deliberately returns
// the string rather than throwing, so that PSIEXCEPTION still records the
// caller's __FILE__/__LINE__ instead of a single spot inside this helper.
std::string cuda_failure_message(const std::string& what, cudaError_t err, const std::string& device_desc = "") {
    std::ostringstream msg;
    msg << what << ": " << cudaGetErrorString(err);

    const std::string hint = cuda_unhealthy_environment_hint(err);
    if (hint.empty()) return msg.str();

    msg << "\n\n"
        << "  This is an unhealthy GPU or node, not a defect in Psi4 or cuEST. No change to the input\n"
        << "  file, keywords, basis set, or Psi4 build will clear it.\n"
        << "\n"
        << "  CUDA status:  " << cudaGetErrorName(err) << " (" << static_cast<int>(err) << ")\n";
    if (!device_desc.empty()) {
        msg << "  Device:       " << device_desc << "\n";
    }
    msg << "\n" << hint << "\n";
    return msg.str();
}

// Identify the device well enough that a user can quote it in a ticket: the
// logical index alone is ambiguous under CUDA_VISIBLE_DEVICES remapping, so
// include the model name and the PCI bus ID that nvidia-smi and the kernel log
// also report.
std::string describe_cuda_device(int device_id, const cudaDeviceProp& props) {
    std::ostringstream desc;
    desc << "index " << device_id << " (" << props.name;
    char bus_id[64] = {0};
    if (cudaDeviceGetPCIBusId(bus_id, static_cast<int>(sizeof(bus_id)), device_id) == cudaSuccess && bus_id[0] != '\0') {
        desc << ", PCI " << bus_id;
    }
    desc << ")";
    return desc.str();
}

void validate_device(int device_id, int device_count) {
    if (device_id < 0 || device_id >= device_count) {
        throw PSIEXCEPTION("cuEST requested, but CUDA reported an invalid active device.");
    }

    cudaDeviceProp props;
    const cudaError_t props_err = cudaGetDeviceProperties(&props, device_id);
    if (props_err != cudaSuccess) {
        throw PSIEXCEPTION(cuda_failure_message("cuEST requested, but cudaGetDeviceProperties failed", props_err));
    }
    if (props.major < 8) {
        std::ostringstream msg;
        msg << "cuEST requires an NVIDIA GPU with compute capability 8.0 or higher; device " << device_id << " ("
            << props.name << ") has compute capability " << props.major << "." << props.minor << ".";
        throw PSIEXCEPTION(msg.str());
    }
}

void configure_runtime() {
    if (runtime_configured) return;

    int device_count = 0;
    const cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess) {
        throw PSIEXCEPTION(cuda_failure_message("cuEST requested, but CUDA device discovery failed", cuda_err));
    }
    if (device_count == 0) {
        throw PSIEXCEPTION("cuEST requested, but no CUDA-capable GPU was found.");
    }

    configured_num_gpus = Process::environment.options.get_int("CUEST_NUM_GPUS");
    if (configured_num_gpus != 1 && configured_num_gpus != 2) {
        throw PSIEXCEPTION("CUEST_NUM_GPUS must be either 1 or 2.");
    }
    if (configured_num_gpus > device_count) {
        throw PSIEXCEPTION("The number of GPUs requested by CUEST_NUM_GPUS exceeds the number available.");
    }

    if (configured_num_gpus == 2) {
        default_device_id = 0;
        vxc_device_id = 1;
    } else {
        cudaError_t current_err = cudaGetDevice(&default_device_id);
        if (current_err != cudaSuccess) {
            throw PSIEXCEPTION(cuda_failure_message("cuEST requested, but cudaGetDevice failed", current_err));
        }
        vxc_device_id = default_device_id;
    }

    validate_device(default_device_id, device_count);
    if (vxc_device_id != default_device_id) validate_device(vxc_device_id, device_count);
    runtime_configured = true;
}

void initialize_context(int device_id, const char* role, cuestHandle_t& cuest, cusolverDnHandle_t& cusolver,
                        cublasHandle_t& cublas, cudaStream_t& stream) {
    if (cuest != 0) return;
    if (stream != 0 || cublas != 0 || cusolver != 0) {
        throw PSIEXCEPTION("Attempting to initialize a partially active cuEST context.");
    }

    const cudaError_t set_device_err = cudaSetDevice(device_id);
    if (set_device_err != cudaSuccess) {
        throw PSIEXCEPTION(cuda_failure_message("cudaSetDevice failed in cuest_init", set_device_err));
    }

    cudaDeviceProp props;
    const cudaError_t props_err = cudaGetDeviceProperties(&props, device_id);
    if (props_err != cudaSuccess) {
        throw PSIEXCEPTION(cuda_failure_message("cudaGetDeviceProperties failed in cuest_init", props_err));
    }
    outfile->Printf("  cuEST initializing %s context on GPU device %d (%s), compute capability %d.%d\n", role,
                    device_id, props.name, props.major, props.minor);

    cuestHandleParameters_t handle_parameters = nullptr;
    try {
        const cudaError_t stream_err = cudaStreamCreate(&stream);
        if (stream_err != cudaSuccess) {
            throw PSIEXCEPTION(cuda_failure_message("cudaStreamCreate failed in cuest_init", stream_err,
                                                     describe_cuda_device(device_id, props)));
        }
        const cublasStatus_t cublas_status = cublasCreate(&cublas);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            throw PSIEXCEPTION("cublasCreate failed in cuest_init");
        }
        const cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            throw PSIEXCEPTION("cusolverDnCreate failed in cuest_init");
        }
        if (cublasSetStream(cublas, stream) != CUBLAS_STATUS_SUCCESS) {
            throw PSIEXCEPTION("cublasSetStream failed in cuest_init");
        }
        if (cusolverDnSetStream(cusolver, stream) != CUSOLVER_STATUS_SUCCESS) {
            throw PSIEXCEPTION("cusolverDnSetStream failed in cuest_init");
        }
        CHECK_CUEST(cuestParametersCreate(CUEST_HANDLE_PARAMETERS, &handle_parameters));
        CHECK_CUEST(cuestParametersConfigure(
            CUEST_HANDLE_PARAMETERS,
            handle_parameters,
            CUEST_HANDLE_PARAMETERS_CUDASTREAM,
            &stream,
            sizeof(stream)
        ));
        CHECK_CUEST(cuestParametersConfigure(
            CUEST_HANDLE_PARAMETERS,
            handle_parameters,
            CUEST_HANDLE_PARAMETERS_CUBLAS,
            &cublas,
            sizeof(cublas)
        ));
        CHECK_CUEST(cuestParametersConfigure(
            CUEST_HANDLE_PARAMETERS,
            handle_parameters,
            CUEST_HANDLE_PARAMETERS_CUSOLVER,
            &cusolver,
            sizeof(cusolver)
        ));
        CHECK_CUEST(cuestCreate(handle_parameters, &cuest));
        CHECK_CUEST(cuestParametersDestroy(CUEST_HANDLE_PARAMETERS, handle_parameters));
        handle_parameters = nullptr;
    } catch (...) {
        if (handle_parameters != nullptr) cuestParametersDestroy(CUEST_HANDLE_PARAMETERS, handle_parameters);
        cuest_cleanup_noexcept(device_id, cuest, cusolver, cublas, stream);
        throw;
    }
}

void select_context(cuest_common::Context context, int& device_id, cuestHandle_t& cuest,
                    cusolverDnHandle_t& cusolver, cublasHandle_t& cublas, cudaStream_t& stream) {
    configure_runtime();
    const bool separate_vxc = context == cuest_common::Context::Vxc && configured_num_gpus == 2;
    if (separate_vxc) {
        device_id = vxc_device_id;
        initialize_context(device_id, "Vxc", vxc_cuest_handle, vxc_cusolver_handle, vxc_cublas_handle,
                           vxc_stream_handle);
        cuest = vxc_cuest_handle;
        cusolver = vxc_cusolver_handle;
        cublas = vxc_cublas_handle;
        stream = vxc_stream_handle;
    } else {
        device_id = default_device_id;
        initialize_context(device_id, "default", cuest_handle, cusolver_handle, cublas_handle, stream_handle);
        cuest = cuest_handle;
        cusolver = cusolver_handle;
        cublas = cublas_handle;
        stream = stream_handle;
    }
}

}  // namespace

namespace psi {

namespace cuest_common {

ScopedContext::ScopedContext(Context context) {
    try {
        std::lock_guard<std::mutex> lock(initialization_mutex);
        configure_runtime();
        const cudaError_t current_err = cudaGetDevice(&previous_device_);
        if (current_err != cudaSuccess) {
            throw PSIEXCEPTION(cuda_failure_message("cuEST requested, but cudaGetDevice failed", current_err));
        }
        select_context(context, device_, cuest_, cusolver_, cublas_, stream_);
        const cudaError_t set_device_err = cudaSetDevice(device_);
        if (set_device_err != cudaSuccess) {
            throw PSIEXCEPTION(cuda_failure_message("cudaSetDevice failed while selecting a cuEST context",
                                                    set_device_err));
        }
    } catch (...) {
        if (previous_device_ >= 0) cudaSetDevice(previous_device_);
        throw;
    }
}

ScopedContext::~ScopedContext() noexcept {
    if (previous_device_ >= 0 && previous_device_ != device_) cudaSetDevice(previous_device_);
}

// Declared in libfock/cuESTCommon.h and called from every cuEST entry point.
void ensure_cuest_initialized() {
    std::lock_guard<std::mutex> lock(initialization_mutex);
    int device_id;
    cuestHandle_t cuest;
    cusolverDnHandle_t cusolver;
    cublasHandle_t cublas;
    cudaStream_t stream;
    select_context(Context::Default, device_id, cuest, cusolver, cublas, stream);
    cudaSetDevice(device_id);
}

}  // namespace cuest_common

namespace cuest_runtime {

void shutdown() {
    std::lock_guard<std::mutex> lock(initialization_mutex);
    cuest_cleanup_noexcept(vxc_device_id, vxc_cuest_handle, vxc_cusolver_handle, vxc_cublas_handle,
                           vxc_stream_handle);
    cuest_cleanup_noexcept(default_device_id, cuest_handle, cusolver_handle, cublas_handle, stream_handle);
}

}  // namespace cuest_runtime
}  // namespace psi

#endif  // USING_cuEST
