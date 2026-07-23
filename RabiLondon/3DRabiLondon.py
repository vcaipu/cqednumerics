''''
This script is used to run the 3D Rabi London solver.
Use: 
export JAX_PLATFORMS=cpu 
To set cpu. 
'''


# Import the FEMSystem Class from directory above
import sys
from pathlib import Path
from VisualizeVF import VisualizeVF
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from FEMSystem import FEMSystem

import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Units2D import Units2D
import os
import time
import resource


from scipy.sparse.linalg import spsolve, minres
from skfem import  ElementTriN1, ElementTriP1, ElementTetN1, ElementTetP1, Basis, BilinearForm, LinearForm, asm, condense, ElementVector
from skfem.helpers import curl, dot, div, grad
import numpy as np
from FEMSystem import FEMSystem
import jax.numpy as jnp
import pickle
from scipy.linalg import expm
from tqdm.auto import tqdm
from scipy.sparse import bmat
from RabiLondonSystem import RabiLondonSystem
try:
    import psutil
except ImportError:
    psutil = None







'''
######### ----- IMPORTANT: INPUT AND OUTPUT FILES ------- ############
'''

# INPUT PICKLE (FROM 3D FEM SOLVER)
input_pickle_file = REPO_ROOT / "3D" / "allplots" / "rl1" / "results.pkl"

# OUTPUT PICKLE
output_pickle_file = "3DRLfinal.pkl" # Same dir 

'''
######### ----- IMPORTANT: INPUT AND OUTPUT FILES ------- ############
'''











### TRUNCATE COEFF Vector such that N is less than density * volume
def truncate_charge_basis_center(pickled_obj, n_keep):
    n0 = int(pickled_obj["n"])
    n_keep = int(n_keep)
    if n_keep <= 0 or n_keep > n0:
        raise ValueError(f"n_keep must be in [1, {n0}]")

    # center window
    start = (n0 - n_keep) // 2
    end = start + n_keep

    # update basis size
    pickled_obj["n"] = n_keep

    # keep coeffs consistent if present
    coeffs = np.asarray(pickled_obj["coeffs"])
    if coeffs.shape[0] == n0:
        pickled_obj["coeffs"] = coeffs[start:end]
    else:
        # if coeffs has unexpected shape, don't silently mismatch
        raise ValueError(f"coeffs length {coeffs.shape[0]} != n ({n0})")
    
    print(f"**** Truncated Coeffs Vector to {len(pickled_obj['coeffs'])} elements!!!! ******")

    return pickled_obj


pickled_obj = {}
with open(input_pickle_file, 'rb') as f:
# with open('./../2D/allplots/rabilondonfine11/results.pkl', 'rb') as f:
#with open('./../2D/allplots/square20sep20num100/results.pkl', 'rb') as f:
    pickled_obj = pickle.load(f)


# Do the Truncation of the Coeffs Vector
pickled_obj = truncate_charge_basis_center(pickled_obj, 20)

# Solver knobs for harder 3D datasets (e.g., sepsweep3/sep1).
MINRES_RTOL = 1e-9
MINRES_MAXITER = 1000000
HELMHOLTZ_EPSILON = 1e-3

rabilondon = RabiLondonSystem(
    pickled_obj,
    minres_rtol=MINRES_RTOL,
    minres_maxiter=MINRES_MAXITER,
    minres_shift=0.0,
    minres_check_convergence=True,
    minres_verbose=True,
)
print(pickled_obj["parameters"])
print(
    f"[run-config] input_pickle_file={input_pickle_file} | "
    f"minres_rtol={MINRES_RTOL} | minres_maxiter={MINRES_MAXITER} | "
    f"helmholtz_epsilon={HELMHOLTZ_EPSILON}"
)

# Now define all physical and material parameters. Only need \xi and desired kapp
xi = .39e-10 * 1
kappa = 0.1
units2d = Units2D(xi,kappa)
# Check units, if transition frequency in nondim units is 4, then must equal plasma frequency
kappa_computed,c_bar = units2d.kappa_cbar_from_xi(xi)
print(f"Computed kappa: {kappa_computed}, c_bar: {c_bar}")
plasma_freq1,plasma_freq2 = c_bar/kappa, units2d.convert_energy_to_rad_s(4,xi)
print("\n\n")
print(f"Using xi = {xi*1e10} A")
print(f"Units check: Plasma frequencies must be equal (in rad/s): {plasma_freq1} | {plasma_freq2}")
print(f"kappa: {kappa}, c_bar: {c_bar}")

n_s  = units2d.n_sfromxi(xi)
print(f"n_s = {n_s} | n_s xi^3: {n_s*xi**3}")
# deltaZ = units2d.compute_dz(n_s,rabilondon.N,rabilondon.area)
# print(f"deltaZ: {deltaZ}")


# London penetration depth from n_s
lambda_L = units2d.london_from_n_s(n_s)
print(f"lambda_L: {lambda_L}")


# Define all Parameters
omega_d_nondim = rabilondon.omega_q # on resonance drive
omega_d_rad_s = units2d.convert_energy_to_rad_s(omega_d_nondim,xi)   #convert to rad/s
print(omega_d_rad_s/c_bar)
print(f"Qubit Frequency (in nondim units): {omega_d_nondim} | (in rad/s):{omega_d_rad_s}")

# Compute expected effective london penetration depth
radicand = 1 / kappa**2 - omega_d_rad_s**2 / c_bar**2
# Allow evanescent regime (negative radicand) by using complex sqrt.
wavevec_mag = np.lib.scimath.sqrt(radicand)
lambda_eff = 1/wavevec_mag
print(f"Expected effective london penetration depth: {lambda_eff}")

source_coord = (0,15,0)
sigma = 15
m = 1 # Magnitude of drive.
source_obj = {
    "source_coord": source_coord,
    "sigma": sigma,
    "m": m
}

timers = {"t0": time.perf_counter(), "prev": None}
timers["prev"] = timers["t0"]

def log_stage_timing(label):
    now = time.perf_counter()
    dt = now - timers["prev"]
    total = now - timers["t0"]
    print(f"[timing-main] {label:<38} {dt:8.3f} s | total {total:8.3f} s")
    timers["prev"] = now

# Run Solver
A_sol, basis_edge = rabilondon.helmholtz_solver(
    omega_d_rad_s,
    c_bar,
    kappa,
    source_obj,
    epsilon=HELMHOLTZ_EPSILON,
)
if not np.all(np.isfinite(A_sol)):
    raise RuntimeError(
        "Helmholtz solve returned non-finite values in A_sol. "
        "Try increasing HELMHOLTZ_EPSILON and/or loosening MINRES_RTOL."
    )
log_stage_timing("helmholtz solve")

# Get Rabi Hamiltonian

print("Checkpoint 1")
H_of_t, A1_mat, A2_mat = rabilondon.rabi_hamiltonian(basis_edge,A_sol,omega_d_nondim)
print("Checkpoint 2")
log_stage_timing("build first time-dependent Hamiltonian")

def _safe_ratio(num, den):
    den_f = float(np.abs(den))
    if den_f == 0.0:
        return np.inf
    return float(num / den_f)

# Get 2 x 2 Matrices for TLS
A1_mat_TLS = rabilondon.TLS_matrix(A1_mat)
A2_mat_TLS = rabilondon.TLS_matrix(A2_mat)
print(f"A1_mat_TLS: {A1_mat_TLS}")
print(f"A2_mat_TLS: {A2_mat_TLS}")
log_stage_timing("first TLS projection/decomposition prep")

# Get Pauli Decomposition
A1_mat_TLS_decomposed = rabilondon.pauli_decompose(A1_mat_TLS)
A2_mat_TLS_decomposed = rabilondon.pauli_decompose(A2_mat_TLS)
print(f"Order is identity, sigma_x, sigma_y, sigma_z")
print(f"A_1 Decomposed: {A1_mat_TLS_decomposed}")
print(f"A_2 Decomposed: {A2_mat_TLS_decomposed}")
a1_off_diag_max, a2_off_diag_max = np.max(np.abs(A1_mat_TLS - np.diag(np.diag(A1_mat_TLS)))), np.max(np.abs(A2_mat_TLS - np.diag(np.diag(A2_mat_TLS))))     
detuning_max = max(np.abs(A1_mat_TLS_decomposed[3]), np.abs(A2_mat_TLS_decomposed[3]))
a2z_detuning = np.abs(A2_mat_TLS_decomposed[3])
print(f"A1 Off Diagonal Max: {a1_off_diag_max}")
print(f"A2 Off Diagonal Max: {a2_off_diag_max}\n")
print(f"Check that A1 off diagonal max is much much larger for Rabi Oscillations!! It is {_safe_ratio(a1_off_diag_max, a2_off_diag_max)} times larger.")
if np.abs(detuning_max) == 0.0:
    print("Check that detuning is much lower too! Detuning proxy (max of A1z/A2z) is exactly zero in this run, so the ratio is undefined/infinite.\n\n")
else:
    print(f"Check that detuning is much lower too! Off diagonal of A1 is {_safe_ratio(a1_off_diag_max, detuning_max)} times larger than the detuning (A1z and A2z).\n\n")

rabi_freq = np.sqrt(np.abs(A1_mat_TLS_decomposed[1])**2 + np.abs(A1_mat_TLS_decomposed[2])**2)  
rabi_period = float(2*np.pi/rabi_freq)
drive_period = float(2*np.pi/omega_d_nondim) # Remember Hamiltonian is nondimmed, so 1/energy or 1/time is nondimmed
print(f"Approx Rabi Frequency (Assuming x and y components of A2 are trivial): {rabi_freq}")
print(f"Rabi Period (nondim): {rabi_period} s")
print(f"Drive Period (nondim): {drive_period} s")
print(f"Iterations: {rabi_period/drive_period*30}\n\n")

print(f"Direct Detuning from A2Z Only (Since A2 has DC term): {a2z_detuning} | Fixing Drive Frequency by Adding Detuning")
omega_d_nondim_new = omega_d_nondim + 2*a2z_detuning

# find first and second eigenvalues and difference  
eigenvalues = np.linalg.eigvalsh(H_of_t(0))
ground_excited_diff = np.abs(eigenvalues[0] - eigenvalues[1])
print(f"Ground to excited state at t=0: {ground_excited_diff}")
print(f"New omega_d_nondim_new: {omega_d_nondim_new}")
print(f"Old detuning: {ground_excited_diff - omega_d_nondim} | 2A2Z is: {2*a2z_detuning}")
print(f"New detuning: {ground_excited_diff - omega_d_nondim_new}")

with open(output_pickle_file, 'wb') as f:
    pickle_obj = {
        "A_sol": A_sol,
        "rabilondon": rabilondon,
        "basis_edge": basis_edge,
        "source_obj": source_obj,
        "omega_d_rad_s": omega_d_rad_s,
        "c_bar": c_bar,
        "kappa": kappa,
        # Don't pickle H_of_t_new directly: it is a local closure and not picklable.
        # Save reconstructable ingredients instead.
        "hamiltonian_time_data": {
            "omega_d_nondim": float(omega_d_nondim_new),
            "static_hamiltonian": np.asarray(rabilondon.hamiltonian),
            "A1_mat": np.asarray(A1_mat),
            "A2_mat": np.asarray(A2_mat),
        },
        'input_pickle_file': input_pickle_file,
        'parameters': pickled_obj["parameters"],
    }
    pickle.dump(pickle_obj, f)
log_stage_timing("checkpoint pickle save")
print(f"[timing-main] total script time so far: {time.perf_counter() - timers['t0']:.3f} s")