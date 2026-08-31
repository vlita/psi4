import json
import os
import subprocess
import sys

import psi4
import pytest

from addons import uusing


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.dft]


def _visible_gpu_count():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        return len([device for device in cuda_visible_devices.split(",") if device.strip() and device.strip() != "-1"])

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0

    return len([line for line in completed.stdout.splitlines() if line.strip()])


_RUN_CUEST_ENERGY = r'''
import json
import sys

import psi4


method = sys.argv[1]
num_gpus = int(sys.argv[2])
result_path = sys.argv[3]
output_path = sys.argv[4]

psi4.core.set_num_threads(4)
psi4.set_memory("2 GiB")
psi4.set_output_file(output_path, False)

molecule = psi4.geometry(
    """
    0 1
    O  0.000000000000  0.000000000000  0.000000000000
    H  0.000000000000 -0.757160000000  0.586260000000
    H  0.000000000000  0.757160000000  0.586260000000
    units angstrom
    symmetry c1
    """
)

options = {
    "basis": "aug-cc-pvdz",
    "scf_type": "df",
    "df_basis_scf": "aug-cc-pvdz-jkfit",
    "puream": True,
    "reference": "rhf",
    "use_cuest": True,
    "cuest_mixed_precision": False,
    "dft_nuclear_scheme": "stratmann",
    "dft_radial_points": 100,
    "dft_spherical_points": 590,
    "e_convergence": 10,
    "d_convergence": 9,
    "maxiter": 200,
}
if num_gpus == 2:
    options["cuest_num_gpus"] = 2

psi4.set_options(options)
energy = psi4.energy(method, molecule=molecule)

with open(result_path, "w") as handle:
    json.dump({"energy": energy}, handle)
'''


def _run_cuest_energy(tmp_path, method, num_gpus):
    stem = f"{method.lower()}_{num_gpus}gpu"
    result_path = tmp_path / f"{stem}.json"
    output_path = tmp_path / f"{stem}.out"

    subprocess.run(
        [sys.executable, "-c", _RUN_CUEST_ENERGY, method, str(num_gpus), str(result_path), str(output_path)],
        check=True,
    )

    with result_path.open() as handle:
        energy = json.load(handle)["energy"]
    return energy, output_path.read_text()


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.skipif(_visible_gpu_count() < 2, reason="cuEST multi-GPU test requires two visible NVIDIA GPUs")
@pytest.mark.parametrize("method", ["b3lyp", "wb97x-v"])
def test_cuest_two_gpu_vxc_jk(method, tmp_path):
    """Two-GPU Vxc/JK execution must reproduce the default one-GPU cuEST path."""
    one_gpu_energy, one_gpu_output = _run_cuest_energy(tmp_path, method, 1)
    two_gpu_energy, two_gpu_output = _run_cuest_energy(tmp_path, method, 2)

    assert psi4.compare_values(
        one_gpu_energy,
        two_gpu_energy,
        atol=1.0e-8,
        label=f"{method} one-GPU versus two-GPU cuEST energy",
    )

    assert "cuEST initializing default context" in one_gpu_output
    assert "cuEST initializing Vxc context" not in one_gpu_output
    assert "cuEST initializing default context on GPU device 0" in two_gpu_output
    assert "cuEST initializing Vxc context on GPU device 1" in two_gpu_output
