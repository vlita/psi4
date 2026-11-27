"""End-to-end check of the cuEST DF-MP2 gradient (cuEST SCF + GPU MP2 gradient)
against plain CPU Psi4 (CPU SCF + CPU DF-MP2 gradient) on a parallel-displaced
benzene dimer.
"""
import math

import psi4
import pytest

from utils import *
from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api]


def _benzene(shift):
    """Idealized D6h benzene (C-C 1.39 A, C-H 1.09 A) in the xy plane, translated by shift."""
    atoms = []
    for k in range(6):
        theta = math.pi / 3.0 * k
        for symbol, r in (("C", 1.39), ("H", 1.39 + 1.09)):
            atoms.append((symbol, r * math.cos(theta) + shift[0], r * math.sin(theta) + shift[1], shift[2]))
    return atoms


# Parallel-displaced dimer: the second ring sits 3.4 A above the first and is slipped 1.6 A along x.
# This is a constructed geometry (not the S22 reference structure); it only needs to be fixed and asymmetric
# enough that every gradient component is nontrivial.
_GEOM = "0 1\n" + "\n".join(
    f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}"
    for s, x, y, z in _benzene((0.0, 0.0, 0.0)) + _benzene((1.6, 0.0, 3.4))
) + """
units angstrom
symmetry c1
no_reorient
no_com
"""

_ATOL_E = 1e-7
_ATOL_G = 1e-6


def _run_mp2_gradient(use_cuest, freeze_core, num_frozen_uocc):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.set_num_threads(4)
    psi4.geometry(_GEOM)
    psi4.set_options({
        "basis": "def2-svp",
        "df_basis_scf": "def2-universal-jkfit",
        "df_basis_mp2": "def2-svp-ri",
        "scf_type": "df",
        "mp2_type": "df",
        "qc_module": "dfmp2",
        "reference": "rhf",
        "freeze_core": freeze_core,
        "num_frozen_uocc": num_frozen_uocc,
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
        # tighten the z-vector solve on both paths so the comparison is not limited by it
        "solver_convergence": 1e-9,
        "use_cuest": use_cuest,
        "cuest_mixed_precision": False,
    })
    grad = psi4.gradient("mp2")
    return {
        "scf": psi4.variable("SCF TOTAL ENERGY"),
        "ss": psi4.variable("MP2 SAME-SPIN CORRELATION ENERGY"),
        "os": psi4.variable("MP2 OPPOSITE-SPIN CORRELATION ENERGY"),
        "total": psi4.variable("MP2 TOTAL ENERGY"),
        "grad": grad.np.copy(),
    }


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.long
@pytest.mark.parametrize("freeze_core,num_frozen_uocc", [
    (False, 0),
    (True, 0),
    (True, 10),
], ids=["all_electron", "frozen_core", "frozen_core_frozen_virtuals"])
def test_cuest_dfmp2_gradient_benzene_pd_dimer(freeze_core, num_frozen_uocc):
    cpu = _run_mp2_gradient(use_cuest=False, freeze_core=freeze_core, num_frozen_uocc=num_frozen_uocc)
    gpu = _run_mp2_gradient(use_cuest=True, freeze_core=freeze_core, num_frozen_uocc=num_frozen_uocc)

    # Sanity: a real MP2 calculation happened, and the gradient is not trivially zero.
    assert cpu["os"] < -1e-1 and cpu["ss"] < -1e-2
    assert abs(cpu["grad"]).max() > 1e-3

    assert compare_values(cpu["scf"], gpu["scf"], _ATOL_E, "SCF energy: cuEST vs CPU")
    assert compare_values(cpu["ss"], gpu["ss"], _ATOL_E, "MP2 SS: cuEST vs CPU")
    assert compare_values(cpu["os"], gpu["os"], _ATOL_E, "MP2 OS: cuEST vs CPU")
    assert compare_values(cpu["total"], gpu["total"], _ATOL_E, "MP2 total: cuEST vs CPU")
    assert compare_values(cpu["grad"], gpu["grad"], _ATOL_G, "MP2 gradient: cuEST vs CPU")
