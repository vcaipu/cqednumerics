# Import the FEMSystem Class from directory above
import sys
from VisualizeVF import VisualizeVF
sys.path.append('..')
from FEMSystem import FEMSystem

import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Units2D import Units2D
from RabiLondon2D import RabiLondon2D
import pickle
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--savefile", type=str, help="File Name to Save it to. Default is rabilondonsave.pkl",default="rabilondonsave.pkl")
parser.add_argument("--readfile", type=str, help="File Name to Read from. Default is None",default=None)

parser.add_argument("--kappa", type=float, help="kappa. Default is 3",default=3)
parser.add_argument("--xi", type=float, help="xi. Default is 0.39e-10",default=0.39e-10)

parser.add_argument(
    "--source_coord",
    type=float,
    nargs=2,
    metavar=("X", "Y"),
    help="source_coord as two numbers: --source_coord X Y. Default is 0 40",
    default=(0.0, 40.0),
)
parser.add_argument("--sigma", type=float, help="sigma. Default is 10",default=10)
parser.add_argument("--m", type=float, help="m. Default is 100",default=100)

parser.add_argument("--num_rabi_periods", type=float, help="num_rabi_periods. Default is 0.5",default=0.5)
parser.add_argument("--steps_per_drive_period", type=int, help="steps_per_drive_period. Default is 5",default=5)    

args = parser.parse_args()

savefile = args.savefile
readfile = args.readfile
kappa = args.kappa
xi = args.xi
source_coord = tuple(args.source_coord)
sigma = args.sigma
m = args.m
num_rabi_periods = args.num_rabi_periods
steps_per_drive_period = args.steps_per_drive_period


'''
Step 1: Define System and All Units
'''
print("========Step 1: Define System and All Units =========")
pickled_obj = {}
#./../2D/allplots/rabilondonfine11/results.pkl
with open(f'{readfile}', 'rb') as f:
# with open('./../2D/allplots/square20sep20num100/results.pkl', 'rb') as f:
    pickled_obj = pickle.load(f)
rabilondon = RabiLondon2D(pickled_obj)
print(pickled_obj["parameters"])

# Now define all physical and material parameters. Only need \xi and desired kappa
xi = .39e-10 * 1
kappa = 3
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
deltaZ = units2d.compute_dz(n_s,rabilondon.N,rabilondon.area)
print(f"deltaZ: {deltaZ}")

# London penetration depth from n_s
lambda_L = units2d.london_from_n_s(n_s)
print(f"lambda_L: {lambda_L}")
print("========Step 1 Finished =========\n\n")

'''
Step 2: Solve Helmholtz Equation
'''
print("========Step 2: Solve Helmholtz Equation =========")
# Define all Parameters
omega_d_nondim = rabilondon.omega_q # on resonance drive
omega_d_rad_s = units2d.convert_energy_to_rad_s(omega_d_nondim,xi)   #convert to rad/s
print(omega_d_rad_s/c_bar)
print(f"Qubit Frequency (in nondim units): {omega_d_nondim} | (in rad/s):{omega_d_rad_s}")

# Compute expected effective london penetration depth
wavevec_mag = np.sqrt(1/kappa**2-omega_d_rad_s**2/c_bar**2)
lambda_eff = 1/wavevec_mag
print(f"Expected effective london penetration depth: {lambda_eff}")

# All from parameters
source_obj = {
    "source_coord": source_coord,
    "sigma": sigma,
    "m": m
}

# Run Solver
A_sol, basis_edge = rabilondon.helmholtz_solver(omega_d_rad_s,c_bar,kappa,source_obj)
print("========Step 2 Finished =========\n\n")

'''
Step 3: Construct Rabi Hamiltonian
'''
print("========Step 3: Construct Rabi Hamiltonian =========")
H_of_t, A1_mat, A2_mat = rabilondon.rabi_hamiltonian(basis_edge,A_sol,omega_d_nondim)

# Get 2 x 2 Matrices for TLS
A1_mat_TLS = rabilondon.TLS_matrix(A1_mat)
A2_mat_TLS = rabilondon.TLS_matrix(A2_mat)
print(f"A1_mat_TLS: {A1_mat_TLS}")
print(f"A2_mat_TLS: {A2_mat_TLS}")

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
print(f"Check that A1 off diagonal max is much much larger for Rabi Oscillations!! It is {a1_off_diag_max/a2_off_diag_max} times larger.")
print(f"Check that detuning is much lower too! Off diagonal of A1 is {a1_off_diag_max/detuning_max} times larger than the detuning (A1z and A2z).\n\n")

rabi_freq = np.sqrt(np.abs(A1_mat_TLS_decomposed[1])**2 + np.abs(A1_mat_TLS_decomposed[2])**2)  
rabi_period = float(2*np.pi/rabi_freq)
drive_period = float(2*np.pi/omega_d_nondim) # Remember Hamiltonian is nondimmed, so 1/energy or 1/time is nondimmed
print(f"Approx Rabi Frequency (Assuming x and y components of A2 are trivial): {rabi_freq}")
print(f"Rabi Period (nondim): {rabi_period} s")
print(f"Drive Period (nondim): {drive_period} s")
print(f"Iterations: {rabi_period/drive_period*30}\n\n")

print(f"Direct Detuning from A2Z Only (Since A2 has DC term): {a2z_detuning} | Fixing Drive Frequency by Adding Detuning")
omega_d_nondim_new = omega_d_nondim - 2*a2z_detuning
H_of_t_new, A1_mat_new, A2_mat_new = rabilondon.rabi_hamiltonian(basis_edge,A_sol,omega_d_nondim_new)
print("========Step 3 Finished =========\n\n")


'''
Step 4: Time Evolution
'''
print("========Step 4: Time Evolution =========")

# Eigenvectors are stored as columns; [:, 0] is the ground-state vector.
c0 = rabilondon.eigenvectors[:, 0]
print(f"Rabi Period: {rabi_period} | Drive Period: {drive_period}")
dt = drive_period/steps_per_drive_period # Time steps on order of drive period (tehcnicallly only need Nyquist sampling rate of drive_period / 2, but 10 per period for safe measure)
end_time = int(rabi_period * num_rabi_periods)
t_grid = np.linspace(0, end_time, int(end_time/dt))   # dt = 0.1
output = rabilondon.evolve_piecewise_progress(c0, t_grid, H_of_t_new)
rabilondon.check_hermiticity_and_norm(H_of_t_new, t_grid, output)
print("========Step 4 Finished =========\n\n")


'''
Step 5: Save Results
'''
print("========Step 5: Save Results =========")
pickled_obj = {
    "rabioutput": output,
    "t_grid": t_grid,
    "A_sol": A_sol,
    "rabilondon": rabilondon,
}
with open(savefile, 'wb') as f:
    pickle.dump(pickled_obj, f)
print("========Step 5 Finished =========\n\n")