"""cuEST DF-MP2 geometry optimization of water from slightly displaced starting points.

The CPU DF-MP2 optimization from the equilibrium-ish geometry gives the reference minimum.
Each cuEST optimization starts a little way off along the PES (symmetric stretch, bend,
asymmetric stretch, ...) and must land on the same minimum.
"""
import numpy as np

import psi4
import pytest

from utils import *
from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api]

# (r(O-H1) [A], r(O-H2) [A], angle(H1-O-H2) [deg])
_START_REF = (0.96, 0.96, 104.5)

_ATOL_E = 1e-7
_ATOL_R = 1e-4  # Angstrom
_ATOL_A = 0.05  # degrees


def _water(r1, r2, a):
    return f"""
0 1
O
H 1 {r1}
H 1 {r2} 2 {a}
symmetry c1
"""


def _internals(mol):
    """O-H1, O-H2 bond lengths (A) and H1-O-H2 angle (deg) of a water molecule."""
    xyz = mol.geometry().np * psi4.constants.bohr2angstroms
    v1, v2 = xyz[1] - xyz[0], xyz[2] - xyz[0]
    r1, r2 = np.linalg.norm(v1), np.linalg.norm(v2)
    a = np.degrees(np.arccos(np.dot(v1, v2) / (r1 * r2)))
    return r1, r2, a


def _optimize(use_cuest, start):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.set_num_threads(4)
    mol = psi4.geometry(_water(*start))
    psi4.set_options({
        "basis": "def2-svp",
        "df_basis_scf": "def2-universal-jkfit",
        "df_basis_mp2": "def2-svp-ri",
        "scf_type": "df",
        "mp2_type": "df",
        "qc_module": "dfmp2",
        "reference": "rhf",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
        "solver_convergence": 1e-9,
        "g_convergence": "gau_tight",
        "geom_maxiter": 50,
        "use_cuest": use_cuest,
        "cuest_mixed_precision": False,
    })
    energy = psi4.optimize("mp2")
    return energy, _internals(mol)


@pytest.fixture(scope="module")
def cpu_minimum():
    return _optimize(use_cuest=False, start=_START_REF)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.long
@pytest.mark.parametrize("start", [
    (1.00, 1.00, 104.5),
    (0.92, 0.92, 104.5),
    (0.96, 0.96, 110.0),
    (0.96, 0.96, 99.0),
    (1.00, 0.93, 104.5),
    (0.99, 0.94, 108.0),
], ids=["stretch", "compress", "open_angle", "close_angle", "asym_stretch", "asym_stretch_bend"])
def test_cuest_dfmp2_opt_water(cpu_minimum, start):
    e_ref, (r1_ref, r2_ref, a_ref) = cpu_minimum
    e_gpu, (r1, r2, a) = _optimize(use_cuest=True, start=start)

    # Sanity: the reference minimum is symmetric and the start really was off it.
    assert compare_values(r1_ref, r2_ref, _ATOL_R, "CPU minimum: r(O-H1) == r(O-H2)")
    assert max(abs(start[0] - r1_ref), abs(start[1] - r2_ref), abs(start[2] - a_ref) / 100.0) > 1e-2

    assert compare_values(e_ref, e_gpu, _ATOL_E, "MP2 energy at minimum: cuEST vs CPU")
    assert compare_values(r1_ref, r1, _ATOL_R, "r(O-H1) at minimum: cuEST vs CPU")
    assert compare_values(r2_ref, r2, _ATOL_R, "r(O-H2) at minimum: cuEST vs CPU")
    assert compare_values(a_ref, a, _ATOL_A, "angle(H-O-H) at minimum: cuEST vs CPU")
