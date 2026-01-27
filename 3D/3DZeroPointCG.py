# Import the FEMSystem Class from directory above
from gmshgen3d import generate_mesh # Before changing directories.
import sys
import os
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

'''
Handle Command Line Args
'''

parser = argparse.ArgumentParser(description="")
parser.add_argument("--plotdir", type=str, help="Directory to save all plots. MUST end with a slash /")
parser.add_argument("--material", type=float, help="n_s\\xi^3, value of material property")
parser.add_argument("--separation", type=float, help="Gap between islands. Default set to 20",default=20.0)

parser.add_argument("--sidelen", type=float, help="sidelength of island. Default set to 20",default=20.0)
parser.add_argument("--gridlen", type=float, help="Length of the outer cube. Default set to 120",default=120.0)
parser.add_argument("--n", type=int, help="Max number difference to be considered, in computational domain. Default is 100",default=100)

parser.add_argument("--lc_large", type=float, help="Element size for large elements. Default set to 10",default=15.0)
parser.add_argument("--lc_small", type=float, help="Element size for small elements. Default set to 1",default=1.5)

args = parser.parse_args()
plotdir = args.plotdir
sidelen = args.sidelen
separation = args.separation
n = args.n # Number of coefficients. NOTE: Just set this to an outside variable. Lots of trouble trying to pass into a dynamical argument, since JAX doesn't like when array indices are dynamical. 
material = args.material #Total number of particles
gridlen = args.gridlen
lc_large = args.lc_large
lc_small = args.lc_small

print(f"RUNNING WITH SEPARATION {separation}")

# Make the plotdirs directory
os.makedirs(plotdir, exist_ok=True)


'''
Part 1: Creating the Mesh
'''
print("Starting Part 1: Creating the Mesh")
# Create the FEMSystem Object
granularity = 30 # number of mesh elements
X = jnp.linspace(0, 1, granularity)
Y = jnp.linspace(0, 1, granularity)
Z = jnp.linspace(0, 1, granularity)

# Define the mesh
# mesh = fem.MeshTet.init_tensor(X,Y,Z)

# Scale and center in 3D
# L = 60.0
# mesh = mesh.scaled(2 * L).translated((-L, -L,-L))

# Generate the Custom Mesh, save to a File
mesh_file_path = f"{plotdir}custommesh.msh"

inner_dim = sidelen / 2
generate_mesh(gridlen, sidelen, separation, inner_dim, lc_large, lc_small, mesh_file_path)

# USING CUSTOM MESH
mesh = fem.Mesh.load(mesh_file_path) 

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

# Step 1: Define the Geometry of two cubic islands:
seps = jnp.arange(1,40,0.1)
int_areas = []


sidelenXY = 25
sidelenZ = 10
centerLeft,centerRight = ((sidelen+separation)/2,0,0), (-(sidelen+separation)/2,0,0)
volume = 2 * (sidelen ** 3)

def theta(x_vec):
    x,y,z = x_vec[0],x_vec[1],x_vec[2]
    cond1 = (jnp.abs(x-centerLeft[0]) <= sidelen / 2) & (jnp.abs(y-centerLeft[1]) <= sidelen / 2) & (jnp.abs(z-centerLeft[2]) <= sidelen / 2)
    cond2 = (jnp.abs(x-centerRight[0]) <= sidelen / 2) & (jnp.abs(y-centerRight[1]) <= sidelen / 2) & (jnp.abs(z-centerRight[2]) <= sidelen / 2)
    return cond1 | cond2

def theta_right_only(x_vec):
    x,y,z= x_vec[0],x_vec[1],x_vec[2]
    cond1 = (jnp.abs(x-centerLeft[0]) <= sidelen / 2) & (jnp.abs(y-centerLeft[1]) <= sidelen / 2) & (jnp.abs(z-centerLeft[2]) <= sidelen / 2)
    return cond1

def smoothed_box(x_vec, center, side_len, sharpness=10.0):
    # Distance from center in each dimension
    dx = jnp.abs(x_vec[0] - center[0]) - side_len / 2
    dy = jnp.abs(x_vec[1] - center[1]) - side_len / 2
    dz = jnp.abs(x_vec[2] - center[2]) - side_len / 2
    
    # Max distance to boundary (positive outside, negative inside)
    dist = jnp.maximum(jnp.maximum(dx, dy), dz)
    
    # Sigmoid maps dist=0 to 0.5. Higher sharpness = steeper transition.
    return jax.nn.sigmoid(-sharpness * dist)

def theta_smoothed(x_vec):
    return smoothed_box(x_vec, centerLeft, sidelen) + smoothed_box(x_vec, centerRight, sidelen)

def theta_right_only_smoothed(x_vec):
    return smoothed_box(x_vec, centerLeft, sidelen)


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
print(f"Time taken to compute Stiffness Matrix and Precompute Potential: {end_time - start_time} seconds")


# 3. Getting Initial Guess
print("Guessing a SINE")
coeff_vector_init = guess_sine(n) / 10

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
temp = objective(initial_guess, A_int, P_int, phi_theta_int)

print("Part 3 Finished: Defined Objective Function")
print("\n\n --------------- \n\n")


'''
Part 4: Run Optimizaton Loop
'''

start_time = time.time()
print("Starting Part 4: Running Optimization Loop")
print("Starting Optimization")
solver = LBFGS(fun=objective,tol=5e-4,verbose=True)
result = solver.run(initial_guess, A_int, P_int, phi_theta_int)
result = result.params 
coeffs,u_interior = unpack(result,n)

end_time = time.time()
print(f"Time taken for optimization loop: {end_time - start_time} seconds")

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

# Pickle the results
pickle_obj = {
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
    "femsystem": femsystem
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

