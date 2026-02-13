# First set to no constant folding 
import os
os.environ["JAX_NO_CONSTANT_FOLD"] = "true"

# Second import the generate_mesh function, before changing directories, but after setting the environment variable above to stop constant folding. 
from gmshgen3d import generate_mesh # Before changing directories.
import sys

# Import the FEMSystem Class from directory above
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from FEMSystem import FEMSystem

# Remaining Imports
import jax.numpy as jnp
import skfem as fem
from jaxopt import LBFGS
from jax.scipy.sparse.linalg import cg
import jax
import matplotlib.pyplot as plt
import argparse
import pickle
import time


totalStartTime = time.time()

'''
Handle Command Line Args
'''

parser = argparse.ArgumentParser(description="")
parser.add_argument("--plotdir", type=str, help="Directory to save all plots. MUST end with a slash /")
parser.add_argument("--material", type=float, help="n_s\\xi^3, value of material property")
parser.add_argument("--separation", type=float, help="Gap between islands. Default set to 20",default=20.0)

parser.add_argument("--sidelenX", type=float, help="sidelength of island, X direction. Default set to 20",default=20.0)
parser.add_argument("--sidelenY", type=float, help="sidelength of island, Y direction. Default set to 20",default=20.0)
parser.add_argument("--sidelenZ", type=float, help="sidelength of island, Z direction. Default set to 20",default=20.0)

parser.add_argument("--padding", type=float, help="Padding between the outer box and the islands. Default set to 10",default=10.0)
parser.add_argument("--n", type=int, help="Max number difference to be considered, in computational domain. Default is 100",default=100)

parser.add_argument("--lc_large", type=float, help="Element size for large elements. Default set to 10",default=10.0)
parser.add_argument("--lc_small", type=float, help="Element size for small elements. Default set to 1",default=1)
parser.add_argument("--intorder", type=int, help="Integration order. Default set to 4",default=4)
parser.add_argument("--opt_tol", type=float, help="LBFGS gradient tolerance. Tighter (e.g. 1e-5) reduces risk of stopping at bad local minima. Default 1e-5", default=1e-5)
parser.add_argument("--opt_maxiter", type=int, help="LBFGS max iterations. Higher (e.g. 1000) gives good runs time to converge. Default 1000", default=1000)
parser.add_argument("--retry_bad_ejec", type=int, help="If EJ/EC is outside [0.01, 10000], retry optimization with different init this many times. Default 1", default=1)

args = parser.parse_args()
plotdir = args.plotdir
sidelenX = args.sidelenX
sidelenY = args.sidelenY
sidelenZ = args.sidelenZ
separation = args.separation
n = args.n # Number of coefficients. NOTE: Just set this to an outside variable. Lots of trouble trying to pass into a dynamical argument, since JAX doesn't like when array indices are dynamical. 
material = args.material #Total number of particles
padding = args.padding
gridlenX = sidelenX + separation + 2 * padding
gridlenY = sidelenY + 2 * padding
gridlenZ = sidelenZ + 2 * padding
lc_large = args.lc_large
lc_small = args.lc_small
intorder = args.intorder
opt_tol = args.opt_tol
opt_maxiter = args.opt_maxiter
retry_bad_ejec = args.retry_bad_ejec

# Physical sanity bounds for EJ/EC (reject solution if outside this range)
EJEC_RATIO_MIN, EJEC_RATIO_MAX = 0.01, 10000.0

print(f"RUNNING WITH THE FOLLOWING PARAMETERS:")
print(f"  plotdir = {plotdir}")
print(f"  material = {material}")
print(f"  separation = {separation}")
print(f"  Island Dimensions: ({sidelenX}, {sidelenY}, {sidelenZ})")
print(f"  Grid Dimensions: ({gridlenX}, {gridlenY}, {gridlenZ}) | Padding: {padding}")
print(f"  n = {n}")
print(f"  lc_large = {lc_large}")
print(f"  lc_small = {lc_small}")
print(f"  opt_tol = {opt_tol}  opt_maxiter = {opt_maxiter}  retry_bad_ejec = {retry_bad_ejec}")

print("\n\n --------------- \n\n")



# Make the plotdirs directory
os.makedirs(plotdir, exist_ok=True)


'''
Part 1: Creating the Mesh
'''
print("Starting Part 1: Creating the Mesh")

# Generate the Custom Mesh, save to a File
mesh_file_path = f"{plotdir}custommesh.msh"

inner_dimX = sidelenX / 2
inner_dimY = sidelenY / 2
inner_dimZ = sidelenZ / 2

generate_mesh(gridlens=(gridlenX, gridlenY, gridlenZ), sidelens=(sidelenX, sidelenY, sidelenZ), inner_dims=(inner_dimX, inner_dimY, inner_dimZ), separation=separation, lc_large=lc_large, lc_small=lc_small, output_file=mesh_file_path)
time1 = time.time() - totalStartTime
print(f"Mesh Generatated | Time: {time1 / 60:.2f} mins {time1% 60:.2f} secs")

# USING CUSTOM MESH
mesh = fem.Mesh.load(mesh_file_path) 
time2 = time.time() - totalStartTime
print(f"Mesh Loaded | Time: {time2 / 60:.2f} mins {time2 % 60:.2f} secs")

# Define the unit Tetrehedral Element
element = fem.ElementTetP1()
intorder = 4

# Now define the FEMSystem
femsystem = FEMSystem(mesh,element,intorder,boundary_condition=0,saveFigsDir=plotdir)
print("Part 1 Finished: Mesh Created")
print(f"Degrees of Freedom: {femsystem.dofs}")
print("\n\n --------------- \n\n")

'''
Part 2: Define Geometry
'''

# Step 1: Define the Geometry of two rectangular islands:
seps = jnp.arange(1,40,0.1)
int_areas = []

centerLeft,centerRight = ((sidelenX+separation)/2,0,0), (-(sidelenX+separation)/2,0,0)
volume = 2 * (sidelenX * sidelenY * sidelenZ)

def theta(x_vec):
    x,y,z = x_vec[0],x_vec[1],x_vec[2]
    cond1 = (jnp.abs(x-centerLeft[0]) <= sidelenX / 2) & (jnp.abs(y-centerLeft[1]) <= sidelenY / 2) & (jnp.abs(z-centerLeft[2]) <= sidelenZ / 2)
    cond2 = (jnp.abs(x-centerRight[0]) <= sidelenX / 2) & (jnp.abs(y-centerRight[1]) <= sidelenY / 2) & (jnp.abs(z-centerRight[2]) <= sidelenZ / 2)
    return cond1 | cond2

def theta_right_only(x_vec):
    x,y,z= x_vec[0],x_vec[1],x_vec[2]
    cond1 = (jnp.abs(x-centerLeft[0]) <= sidelenX / 2) & (jnp.abs(y-centerLeft[1]) <= sidelenY / 2) & (jnp.abs(z-centerLeft[2]) <= sidelenZ / 2)
    return cond1

def smoothed_box(x_vec, center, sx, sy, sz, sharpness=10.0):
    # Distance from center in each dimension
    dx = jnp.abs(x_vec[0] - center[0]) - sx / 2
    dy = jnp.abs(x_vec[1] - center[1]) - sy / 2
    dz = jnp.abs(x_vec[2] - center[2]) - sz / 2
    
    # Max distance to boundary (positive outside, negative inside)
    dist = jnp.maximum(jnp.maximum(dx, dy), dz)
    
    # Sigmoid maps dist=0 to 0.5. Higher sharpness = steeper transition.
    return jax.nn.sigmoid(-sharpness * dist)

def theta_smoothed(x_vec):
    return smoothed_box(x_vec, centerLeft, sidelenX, sidelenY, sidelenZ) + smoothed_box(x_vec, centerRight, sidelenX, sidelenY, sidelenZ)

def theta_right_only_smoothed(x_vec):
    return smoothed_box(x_vec, centerLeft, sidelenX, sidelenY, sidelenZ)


theta_at_dofs = theta(femsystem.doflocs).astype(jnp.float32)
integrated_volume = femsystem.integrate(lambda u,grad_u,x: u,theta_at_dofs)
print(f"Area: {volume} | Integrated Area Estimate: {integrated_volume}")

print("Part 2 Finished: Defined Geometry")
print("\n\n --------------- \n\n")

'''
Part 3: Define Objective Function
'''

# Set constants
N_val = material * integrated_volume # The Value of "N", number of particles, in terms of quantities we know

'''
Helper Functions for Integrals
'''

# Really u*laplacian(u) = -(grad u)^2
def laplacian(u,grad_u,x):
    return -1*jnp.sum(grad_u**2,axis=0)

# For Potential Energy Double Inetgral, u1 is our function phi_{+/-} and u2 is theta. 
def u_squared(u,grad_u,u2,grad_u2,x):
    return u**2

# Define Pre-Computed Values for Theta and Green's Function:
def theta_func(u,grad_u,u2,grad_u2,x):
    return u2

# U_{++++} or U_{----}, Really N * \alpha
def alpha(u,A_int,P_int):
    return 1/(material) * femsystem.double_integral_cg(lambda u1,a,b,c,d: u1**2,lambda u1,a,b,c,d: u1**2,A_int,P_int,u,u)

# U_{+--+} = U{-++-} - Remember middle two are wrt to y, Outer two wrt to x, from notation used in doc
def beta(u1_arg,u2_arg,A_int,P_int):
    return 1/(material) * femsystem.double_integral_cg(lambda u1,a,b,c,d: u1**2, lambda a,b,u2,c,d: u2**2, A_int, P_int, u1_arg,u2_arg)

# U_{++--} = U{+-+-}
def gamma(u1_arg,u2_arg,A_int,P_int):
    return 1/(material) * femsystem.double_integral_cg(lambda u1,a,u2,c,d: u1*u2, lambda u1,b,u2,c,d: u1*u2, A_int, P_int, u1_arg,u2_arg)

'''
Helper Functions for Matrices
'''

# N x N, with k off diagonal all 1s
def off_diag(n,k):
    ones_super, ones_sub = jnp.ones(n - k, dtype=jnp.int32),jnp.ones(n - k, dtype=jnp.int32)
    super_diag_matrix,sub_diag_matrix= jnp.diag(ones_super, k=k),jnp.diag(ones_sub, k=-1*k)
    result = super_diag_matrix + sub_diag_matrix
    return result 

def cos_phi(n):
    return off_diag(n,1) / 2

def cos_2phi(n):
    return off_diag(n,2) / 2

def Jz(n):
    j = (n-1)/2
    diagonals = j - jnp.arange(n)
    return jnp.diag(diagonals)

def Jz2(n):
    j = (n-1)/2
    diagonals = j - jnp.arange(n)
    return jnp.diag(diagonals**2)

def expval(mat,vec):
    return jnp.vdot(vec,mat @ vec)

def normalize_vec(vec):
    norm_v = jnp.linalg.norm(vec)
    normalized_v = jnp.where(jnp.isclose(norm_v, 0.0), vec, vec / norm_v )
    return normalized_v

def guess_gaussian(n,stddevs=4):
    x = jnp.linspace(-stddevs, stddevs, n)
    mu,sigma = 0.0,1.0
    exponent = -jnp.square(x - mu) / (2.0 * jnp.square(sigma))
    gaussian_array = jnp.exp(exponent)
    return gaussian_array

def guess_sine(n):
    """Initial guess using a single sine half-period."""
    x = jnp.linspace(0, jnp.pi, n)
    return jnp.sin(x)

def guess_random_normal(n, key=jax.random.PRNGKey(42), scale=0.1):
    """Initial guess using random values from a normal distribution."""
    return jax.random.normal(key, (n,)) * scale

# get first N as the vector of coeffs, remaining as u_interior
def unpack(vec,n):
    coeff_vec,u = vec[:n],vec[n:]
    return coeff_vec,u


# VERY VERY IMPORTANT TO PASS IN A_int AS AN ARGUMENT, AND SET TO CONSTANT IN OPTIMIZATION LOOP
# This is because when JAX compiles this function, it will treat the A_int as a "tracer", so just any matrix of constants with some shape. 
# If you hardcode it into the function, it will treat it as an actual part of the code and will spent time compiling a massive amount of hardcoded values as "code" essentially. This is why it takes almost 10 minutes to run first optimization iteration. 
# @jax.jit

def epsilon_func(u_global, P_int, phi_theta_int):
    # Kinetic Term
    kinetic = -4 * femsystem.integrate(laplacian,u_global)

    # Potential Term: -2 * <u^2, G theta> = -2 * <u^2, phi_theta>
    u_quad = femsystem._interpolate_values(u_global)
    weighted_u2 = (u_quad**2) * femsystem.weights
    v_u2 = P_int.T @ weighted_u2.ravel()
    potential = -2 * (v_u2 @ phi_theta_int)

    return kinetic  + potential

def E(u_global, A_int, P_int, phi_theta_int):
    return epsilon_func(u_global, P_int, phi_theta_int) + (N_val - 1)*alpha(u_global, A_int, P_int)


'''
Before you start the optimization loop:
1. Define Objective
2. Compute Interaction Kernel
3. Get Initial Guess
'''


# 1. Defining Objective

def ej_ec_e0(u_interior,A_int,P_int,phi_theta_int):
    # Unpack even and odd modes
    u_even, u_odd = femsystem.separate_even_odd_apply_by_and_norm(u_interior)
    
    # Precompute shared terms to minimize CG solves
    gamma_val = gamma(u_even, u_odd, A_int, P_int)
    E_plus = E(u_even, A_int, P_int, phi_theta_int)
    E_minus = E(u_odd, A_int, P_int, phi_theta_int)

    # Construct Objective
    e0 = ( E_plus + E_minus ) / 2 - gamma_val # Full Zero Point Energy

    hz = ( E_plus - E_minus ) 
    lambda_x = 4 * gamma_val

    # Really E_J and E_C per particle (E_J/N, E_C/N)
    E_J = -1*hz / 2
    E_C = lambda_x / (N_val)

    return E_J, E_C, e0

@jax.jit
def objective(vec, A_int, P_int, phi_theta_int):
    # Unpack the modes from the coefficients
    coeff_vec, u_interior = unpack(vec, n)

    # Normalize Coeff Vector: 
    coeff_vec_norm = normalize_vec(coeff_vec)

    # Get E_J, E_C, and e0
    E_J, E_C, e0 = ej_ec_e0(u_interior, A_int, P_int, phi_theta_int)

    # Josephson Tunneling Term
    cos1 = cos_phi(n)
    first_harmonic = (-1* E_J) * expval(cos1, coeff_vec_norm)

    # Capacitive Term
    jz2 = Jz2(n)
    capacitive = E_C * expval(jz2, coeff_vec_norm)

    return capacitive + first_harmonic + e0


# 2. Computing Interaction Kernel
start_time = time.time()
A_int, P_int = femsystem.get_stiffness_matrix()

# Precompute the potential solve for the fixed geometry (theta)
# This removes 2 CG solves from the JIT'd objective function
theta_at_quad = femsystem._interpolate_values(theta_at_dofs)
weighted_theta = theta_at_quad * femsystem.weights
v_theta = P_int.T @ weighted_theta.ravel()
phi_theta_int, _ = cg(A_int, v_theta)

end_time = time.time()
time3 = end_time - start_time
print(f"Time taken to compute Stiffness Matrix and Precompute Potential: {time3} seconds")

# 3. Getting Initial Guess
print("Guessing a Gaussian")
coeff_vector_init = guess_gaussian(n,stddevs=6) / 10

# Plotting Coefficients
x = (n-1)/2 - jnp.arange(n)
fig, ax = plt.subplots(figsize=(8, 6)) # Creates a figure and a single subplot (axes)
ax.plot(x,coeff_vector_init,".")
ax.set_xlabel('Charge Imbalance Eigenvalue')
ax.set_ylabel('Coefficient Value')
femsystem._save_fig(plt.gcf(),"Initial Guess Coefficients")


u_interior_init = femsystem.ones_on_island(theta_right_only_smoothed)
initial_guess = jnp.concatenate((coeff_vector_init, u_interior_init), axis=0)

'''
Testing, for a sanity check, and to do a jit compilation
'''
start_time = time.time()
temp = objective(initial_guess, A_int, P_int, phi_theta_int)
end_time = time.time()
time4 = end_time - start_time
print(f"Time taken to run objective function once, for first time: {time4} seconds")

print("Part 3 Finished: Defined Objective Function")
print("\n\n --------------- \n\n")


'''
Part 4: Run Optimization Loop (with optional retries if EJ/EC is unphysical)
'''

def make_initial_guess(init_type="gaussian"):
    """Build (coeff_vec, u_interior) initial guess. init_type: 'gaussian', 'gaussian_narrow', 'sine'."""
    if init_type == "gaussian":
        c = guess_gaussian(n, stddevs=6) / 10
    elif init_type == "gaussian_narrow":
        c = guess_gaussian(n, stddevs=4) / 10
    elif init_type == "sine":
        c = guess_sine(n) * 0.1
    else:
        c = guess_gaussian(n, stddevs=6) / 10
    u_init = femsystem.ones_on_island(theta_right_only_smoothed)
    return jnp.concatenate((c, u_init), axis=0)

solver = LBFGS(fun=objective, tol=opt_tol, maxiter=opt_maxiter, verbose=True)
init_types = ["gaussian", "gaussian_narrow", "sine"]
best_result = None
best_objective = jnp.inf   # we minimize objective (more negative = better)
best_ejec_ok = False
time5_total = 0.0
attempt = 0
max_attempts = 1 + retry_bad_ejec

while attempt < max_attempts:
    init_type = init_types[attempt % len(init_types)]
    guess = make_initial_guess(init_type)
    print(f"Starting Part 4: Running Optimization Loop (attempt {attempt + 1}/{max_attempts}, init={init_type})")
    start_time = time.time()
    result_cand = solver.run(guess, A_int, P_int, phi_theta_int)
    result_cand = result_cand.params
    time5_total += time.time() - start_time
    coeffs_cand, u_interior_cand = unpack(result_cand, n)
    E_J_cand, E_C_cand, e0_cand = ej_ec_e0(u_interior_cand, A_int, P_int, phi_theta_int)
    ejec_ratio = float(E_J_cand / E_C_cand)
    ejec_ok = (EJEC_RATIO_MIN <= ejec_ratio <= EJEC_RATIO_MAX)
    obj_val = float(objective(result_cand, A_int, P_int, phi_theta_int))
    print(f"  Attempt {attempt + 1}: objective={obj_val:.6f}  EJ/EC={ejec_ratio:.4f}  (physical range: {ejec_ok})")
    if best_result is None or (ejec_ok and not best_ejec_ok) or (ejec_ok == best_ejec_ok and obj_val < best_objective):
        best_result = result_cand
        best_objective = obj_val
        best_ejec_ok = ejec_ok
    if ejec_ok or attempt >= max_attempts - 1:
        break
    print(f"  EJ/EC outside [{EJEC_RATIO_MIN}, {EJEC_RATIO_MAX}]; retrying with init={init_types[(attempt + 1) % len(init_types)]}.")
    attempt += 1

result = best_result
coeffs, u_interior = unpack(result, n)
time5 = time5_total
if not best_ejec_ok:
    print("WARNING: All attempts gave EJ/EC outside physical range; using best objective. Consider increasing retry_bad_ejec or tightening opt_tol.")

print(f"Time taken for optimization loop(s): {time5} seconds")
print("Part 4 Finished: Ran Optimization Loop")
print("\n\n --------------- \n\n")

'''
Part 5: Plot and Visualize Results
'''

# Get Even and Odd Modes
u_even,u_odd = femsystem.separate_even_odd_apply_by_and_norm(u_interior)
u_even_interior,u_odd_interior = u_even[femsystem.interior_dofs],u_odd[femsystem.interior_dofs]

energy = objective(result, A_int, P_int, phi_theta_int)
E_J, E_C, e0 = ej_ec_e0(u_interior, A_int, P_int, phi_theta_int)

print(f"EJ: {E_J} | EC: {E_C} | e0: {e0}")
print(f"EJ/EC RATIO: {E_J/E_C}")

totalEndTime = time.time()
totalTime = totalEndTime - totalStartTime
print(f"Total Time Taken: {totalTime} seconds")

# Pickle the results
pickle_obj = {
    "parameters": {
        "material": material,
        "separation": separation,
        "sidelenX": sidelenX,
        "sidelenY": sidelenY,
        "sidelenZ": sidelenZ,
        "padding": padding,
        "gridlenX": gridlenX,
        "gridlenY": gridlenY,
        "gridlenZ": gridlenZ,
        "lc_large": lc_large,
        "lc_small": lc_small
    },
    "times": {
        "mesh_gen": time1,
        "mesh_load": time2,
        "matrix_gen": time3,
        "objective_run": time4,
        "optimization_loop": time5
    },
    "n": n,
    "E_J": E_J,
    "E_C": E_C,
    "e0": e0,
    "objective": energy, # Final objective value
    "theta_at_dofs": theta_at_dofs,
    "integrated_volume": integrated_volume,
    "coeffs": coeffs,
    "u_even": u_even,
    "u_odd": u_odd,
    "femsystem": femsystem,
    "totalTime": totalTime
}
with open(plotdir+"results.pkl", 'wb') as f:
    pickle.dump(pickle_obj,f)

# Plotting Coefficients
x = (n-1)/2 - jnp.arange(n)
fig, ax = plt.subplots(figsize=(8, 6)) # Creates a figure and a single subplot (axes)
ax.plot(x,coeffs,".")
ax.set_xlabel('Charge Imbalance Eigenvalue')
ax.set_ylabel('Coefficient Value')
femsystem._save_fig(plt.gcf(),"Coefficients")

print("Part 5 Finished: Saving Plots")

