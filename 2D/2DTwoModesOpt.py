# First set to no constant folding 
import os
os.environ["JAX_NO_CONSTANT_FOLD"] = "true"
os.environ["JAX_ENABLE_X64"] = "true"

# Second import the generate_mesh functions, before changing directories, but after setting the
# environment variable above to stop constant folding.
from gmshgen2d import generate_mesh as generate_mesh_rect
from gmshgen2d_composite import generate_mesh as generate_mesh_composite
import sys

# Import the FEMSystem Class from directory above
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from FEMSystem import FEMSystem
from skfem.mesh import MeshTri1, MeshTri2

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


# Only Needed for the Global Optimizer
from scipy.optimize import basinhopping, OptimizeResult
import numpy as np


totalStartTime = time.time()

'''
Handle Command Line Args
'''

parser = argparse.ArgumentParser(description="")
parser.add_argument("--plotdir", type=str, help="Directory to save all plots. MUST end with a slash /")
parser.add_argument("--separation", type=float, help="Gap between islands. Default set to 20",default=20.0)

parser.add_argument("--sidelenX", type=float, help="sidelength of island, X direction. Default set to 20",default=20.0)
parser.add_argument("--sidelenY", type=float, help="sidelength of island, Y direction. Default set to 20",default=20.0)
parser.add_argument("--geometry", type=str, choices=["rect", "composite"], default="rect",
                    help="Island geometry mode. 'rect' = original two rectangles, "
                         "'composite' = reflected 8-sided islands from two touching rectangles.")
parser.add_argument("--sidelen2X", type=float, default=10.0,
                    help="Second rectangle X length for composite geometry.")
parser.add_argument("--sidelen2Y", type=float, default=10.0,
                    help="Second rectangle Y length for composite geometry.")
parser.add_argument("--inner_dim", type=float, default=None,
                    help="Inset distance for composite geometry inner coarse region. "
                         "Default None uses min(sidelens)/4.")

parser.add_argument("--padding", type=float, help="Padding between the outer box and the islands. Default set to 10",default=10.0)
parser.add_argument("--n", type=int, help="Max number difference to be considered, in computational domain. Default is 50",default=50)
parser.add_argument("--N", type=int, help="Total number of particles. Default is 70",default=70)

parser.add_argument("--lc_large", type=float, help="Element size for large elements. Default set to 10",default=10.0)
parser.add_argument("--lc_small", type=float, help="Element size for small elements. Default set to 1",default=1)
parser.add_argument("--intorder", type=int, help="Integration order. Default set to 5",default=5)
parser.add_argument("--opt_tol", type=float, help="LBFGS gradient tolerance. Tighter (e.g. 1e-5) reduces risk of stopping at bad local minima. Default 1e-5", default=1e-5)
parser.add_argument("--opt_maxiter", type=int, help="LBFGS max iterations. Higher (e.g. 1000) gives good runs time to converge. Default 500", default=1000)
parser.add_argument("--element_order", type=int, choices=[1, 2], default=1, help="Finite element order: 1 = linear (P1), 2 = quadratic (P2). Default 1.")
parser.add_argument("--mesh_file", type=str, help="Path to the mesh file. Default is None, which will generate a new mesh.", default=None)

parser.add_argument("--full_lambda_y", type=bool, help="Whether to include the lambda_y term in the objective function. Default is False.", default=False)
parser.add_argument("--coeff_norm_penalty", type=float,
                    help="Small penalty to keep raw coeff norm near 1 (stabilizes LBFGS null direction).",
                    default=1e-6)


args = parser.parse_args()
plotdir = args.plotdir
sidelenX = args.sidelenX
sidelenY = args.sidelenY
geometry = args.geometry
sidelen2X = args.sidelen2X
sidelen2Y = args.sidelen2Y
inner_dim_arg = args.inner_dim
separation = args.separation
n = args.n # Number of coefficients. NOTE: Just set this to an outside variable. Lots of trouble trying to pass into a dynamical argument, since JAX doesn't like when array indices are dynamical. 
N_val = args.N #Total number of particles
padding = args.padding
if geometry == "composite":
    gridlenX = 2 * sidelenX + 2 * sidelen2X + separation + 2 * padding
    gridlenY = max(sidelenY, sidelen2Y) + 2 * padding
else:
    gridlenX = 2 * sidelenX + separation + 2 * padding
    gridlenY = sidelenY + 2 * padding
lc_large = args.lc_large
lc_small = args.lc_small
intorder = args.intorder
opt_tol = args.opt_tol
opt_maxiter = args.opt_maxiter
element_order = args.element_order
mesh_file = args.mesh_file
full_lambda_y = args.full_lambda_y
coeff_norm_penalty = args.coeff_norm_penalty

print(f"RUNNING WITH THE FOLLOWING PARAMETERS:")
print(f"  plotdir = {plotdir}")
print(f"  N_val = {N_val}")
print(f"  separation = {separation}")
print(f"  Island Dimensions: ({sidelenX}, {sidelenY})")
print(f"  geometry = {geometry}")
if geometry == "composite":
    print(f"  Composite Second Dimensions: ({sidelen2X}, {sidelen2Y})")
print(f"  Grid Dimensions: ({gridlenX}, {gridlenY}) | Padding: {padding}")
print(f"  n = {n}")
print(f"  lc_large = {lc_large}")
print(f"  lc_small = {lc_small}")
print(f"  opt_tol = {opt_tol}  opt_maxiter = {opt_maxiter}")
print(f"  coeff_norm_penalty = {coeff_norm_penalty}")
print(f"  element_order = {element_order}  ({'P1 linear' if element_order == 1 else 'P2 quadratic'})")
print(f"  mesh_file = {mesh_file}")
print("\n\n --------------- \n\n")
sys.stdout.flush()  # ensure params appear first in log when stdout is redirected (e.g. runSingle.slurm)

# Make the plotdirs directory
os.makedirs(plotdir, exist_ok=True)

'''
Part 1: Creating the Mesh
'''
print("Starting Part 1: Creating the Mesh")

# Generate the Custom Mesh, save to a File

inner_dimX = sidelenX / 2
inner_dimY = sidelenY / 2
inner_dim = inner_dim_arg
if inner_dim is None:
    inner_dim = min(sidelenX, sidelenY, sidelen2X, sidelen2Y) / 4.0

# Generate Mesh, or use existing mesh file
if mesh_file is None:
    mesh_file = f"{plotdir}custommesh.msh"
    if geometry == "composite":
        generate_mesh_composite(
            gridlens=(gridlenX, gridlenY),
            first_dims=(sidelenX, sidelenY),
            second_dims=(sidelen2X, sidelen2Y),
            inner_dim=inner_dim,
            separation=separation,
            lc_large=lc_large,
            lc_small=lc_small,
            output_file=mesh_file,
            element_order=element_order,
        )
    else:
        generate_mesh_rect(
            gridlens=(gridlenX, gridlenY),
            sidelens=(sidelenX, sidelenY),
            inner_dims=(inner_dimX, inner_dimY),
            separation=separation,
            lc_large=lc_large,
            lc_small=lc_small,
            output_file=mesh_file,
            element_order=element_order,
        )
else: 
    print(f"Using Existing Mesh File: {mesh_file}")

time1 = time.time() - totalStartTime
print(f"Mesh Generatated | Time: {time1 / 60:.2f} mins {time1% 60:.2f} secs")

# Load mesh (MeshTet2 for quadratic elements has 10 nodes per tet)
if element_order == 2:
    mesh = MeshTri2.load(mesh_file)
else:
    mesh = MeshTri1.load(mesh_file)
time2 = time.time() - totalStartTime
print(f"Mesh Loaded | Time: {time2 / 60:.2f} mins {time2 % 60:.2f} secs")

# Define the tetrahedral element: P1 = linear, P2 = quadratic (polynomial)
element = fem.ElementTriP2() if element_order == 2 else fem.ElementTriP1()

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
if geometry == "composite":
    area = 2 * (sidelenX * sidelenY + sidelen2X * sidelen2Y)
else:
    area = 2 * (sidelenX * sidelenY)


def _composite_bounds(center_x):
    # "First" rectangle is centered at center_x. "Second" is attached on the outside.
    first_xmin = center_x - sidelenX / 2
    first_xmax = center_x + sidelenX / 2
    first_ymin = -sidelenY / 2
    first_ymax = sidelenY / 2

    if center_x > 0:
        second_xmin = first_xmax
        second_xmax = second_xmin + sidelen2X
    else:
        second_xmax = first_xmin
        second_xmin = second_xmax - sidelen2X

    second_ymin = -sidelen2Y / 2
    second_ymax = sidelen2Y / 2
    return (
        first_xmin,
        first_xmax,
        first_ymin,
        first_ymax,
        second_xmin,
        second_xmax,
        second_ymin,
        second_ymax,
    )

def theta(x_vec):
    x,y = x_vec[0],x_vec[1]
    if geometry == "composite":
        (
            l1_xmin, l1_xmax, l1_ymin, l1_ymax,
            l2_xmin, l2_xmax, l2_ymin, l2_ymax,
        ) = _composite_bounds(centerLeft[0])
        (
            r1_xmin, r1_xmax, r1_ymin, r1_ymax,
            r2_xmin, r2_xmax, r2_ymin, r2_ymax,
        ) = _composite_bounds(centerRight[0])

        left_first = (x >= l1_xmin) & (x <= l1_xmax) & (y >= l1_ymin) & (y <= l1_ymax)
        left_second = (x >= l2_xmin) & (x <= l2_xmax) & (y >= l2_ymin) & (y <= l2_ymax)
        right_first = (x >= r1_xmin) & (x <= r1_xmax) & (y >= r1_ymin) & (y <= r1_ymax)
        right_second = (x >= r2_xmin) & (x <= r2_xmax) & (y >= r2_ymin) & (y <= r2_ymax)
        return left_first | left_second | right_first | right_second

    cond1 = (jnp.abs(x-centerLeft[0]) <= sidelenX / 2) & (jnp.abs(y-centerLeft[1]) <= sidelenY / 2)
    cond2 = (jnp.abs(x-centerRight[0]) <= sidelenX / 2) & (jnp.abs(y-centerRight[1]) <= sidelenY / 2)
    return cond1 | cond2

def theta_right_only(x_vec):
    x,y = x_vec[0],x_vec[1]
    if geometry == "composite":
        (
            l1_xmin, l1_xmax, l1_ymin, l1_ymax,
            l2_xmin, l2_xmax, l2_ymin, l2_ymax,
        ) = _composite_bounds(centerLeft[0])
        left_first = (x >= l1_xmin) & (x <= l1_xmax) & (y >= l1_ymin) & (y <= l1_ymax)
        left_second = (x >= l2_xmin) & (x <= l2_xmax) & (y >= l2_ymin) & (y <= l2_ymax)
        return left_first | left_second

    cond1 = (jnp.abs(x-centerLeft[0]) <= sidelenX / 2) & (jnp.abs(y-centerLeft[1]) <= sidelenY / 2)
    return cond1

def smoothed_box(x_vec, center, sx, sy, sharpness=10.0):
    # Distance from center in each dimension
    dx = jnp.abs(x_vec[0] - center[0]) - sx / 2
    dy = jnp.abs(x_vec[1] - center[1]) - sy / 2
    
    # Max distance to boundary (positive outside, negative inside)
    dist = jnp.maximum(dx, dy)
    
    # Sigmoid maps dist=0 to 0.5. Higher sharpness = steeper transition.
    return jax.nn.sigmoid(-sharpness * dist)

def theta_smoothed(x_vec):
    if geometry == "composite":
        second_center_left = (centerLeft[0] + (sidelenX + sidelen2X) / 2.0, 0.0, 0.0)
        second_center_right = (centerRight[0] - (sidelenX + sidelen2X) / 2.0, 0.0, 0.0)
        left_val = smoothed_box(x_vec, centerLeft, sidelenX, sidelenY) + smoothed_box(
            x_vec, second_center_left, sidelen2X, sidelen2Y
        )
        right_val = smoothed_box(x_vec, centerRight, sidelenX, sidelenY) + smoothed_box(
            x_vec, second_center_right, sidelen2X, sidelen2Y
        )
        return left_val + right_val

    return smoothed_box(x_vec, centerLeft, sidelenX, sidelenY) + smoothed_box(
        x_vec, centerRight, sidelenX, sidelenY
    )

def theta_right_only_smoothed(x_vec):
    if geometry == "composite":
        second_center_left = (centerLeft[0] + (sidelenX + sidelen2X) / 2.0, 0.0, 0.0)
        return smoothed_box(x_vec, centerLeft, sidelenX, sidelenY) + smoothed_box(
            x_vec, second_center_left, sidelen2X, sidelen2Y
        )

    return smoothed_box(x_vec, centerLeft, sidelenX, sidelenY)

theta_at_dofs = theta(femsystem.doflocs).astype(jnp.float32)
theta_smoothed_at_dofs = theta_smoothed(femsystem.doflocs).astype(jnp.float32)
integrated_area = femsystem.integrate(lambda u,grad_u,x: u,theta_at_dofs)
print(f"Area: {area} | Integrated Area Estimate: {integrated_area}")

print("Part 2 Finished: Defined Geometry")
print("\n\n --------------- \n\n")

'''
Part 3: Define Objective Function
'''

material = N_val / integrated_area
# So 1/Material gives the correct coefficient.

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

def lambda_y(u1_arg,u2_arg,A_int,P_int):
    return 2*beta(u1_arg,u2_arg,A_int,P_int) - alpha(u1_arg,A_int,P_int) - alpha(u2_arg,A_int,P_int)

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
    gaussian_array /= jnp.linalg.norm(gaussian_array)
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
def ej_ec_e0(u_right_interior,A_int,P_int,phi_theta_int):
    # Unpack even and odd modes
    u_even, u_odd = femsystem.get_even_odd_modes(u_right_interior)
    
    # Precompute shared terms to minimize CG solves
    gamma_val = gamma(u_even, u_odd, A_int, P_int)
    E_plus = E(u_even, A_int, P_int, phi_theta_int)
    E_minus = E(u_odd, A_int, P_int, phi_theta_int)

    # Construct Objective
    e0 = ( E_plus + E_minus ) / 2 - gamma_val # Full Zero Point Energy
    e0 *= N_val

    hz = ( E_plus - E_minus ) 
    lambda_x = 4 * gamma_val

    # Really E_J and E_C per particle (E_J/N, E_C/N)
    E_J = -1*hz / 2 * N_val
    E_C = lambda_x

    lambda_y_val = 0.0
    if full_lambda_y:
        lambda_y_val = lambda_y(u_even, u_odd, A_int, P_int)
    
    return E_J, E_C, e0, lambda_y_val

@jax.jit
def objective(vec, A_int, P_int, phi_theta_int):
    # Unpack the modes from the coefficients
    coeff_vec, u_right_interior = unpack(vec, n)

    # Normalize Coeff Vector: 
    coeff_vec_norm = normalize_vec(coeff_vec)

    E_J, E_C, e0, lambda_y_val = ej_ec_e0(u_right_interior, A_int, P_int, phi_theta_int)

    # Get E_J, E_C, and e0
    if full_lambda_y:
        # 1. E_C correction
        E_C -= lambda_y_val/2

        # 2. Second Harmonic Correction
        cos2 = cos_2phi(n)
        second_harmonic = -1*lambda_y_val/2 * (N_val/2)*(N_val/2+1)* expval(cos2, coeff_vec_norm)
        e0 += second_harmonic

        # 3. ZPE Correction
        zpe_correction = lambda_y_val /2 * (N_val/2)**2
        e0 += zpe_correction

        # Correction to e0 to account for the lambda_y term
        e0 -= lambda_y_val /4 * N_val 

    # Josephson Tunneling Term
    cos1 = cos_phi(n)
    first_harmonic = (-1* E_J) * expval(cos1, coeff_vec_norm)

    # Capacitive Term
    jz2 = Jz2(n)
    capacitive = E_C * expval(jz2, coeff_vec_norm)

    # Objective is scale-invariant in coeff direction; tiny radial penalty
    # removes the null direction so LBFGS doesn't blow up raw coeff magnitudes.
    coeff_norm = jnp.linalg.norm(coeff_vec)
    radial_penalty = coeff_norm_penalty * (coeff_norm - 1.0) ** 2
    return capacitive + first_harmonic + e0 + radial_penalty


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
coeff_vector_init = guess_gaussian(n,stddevs=30)

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
print(f"Starting Part 4: Running Optimization Loop")

def create_lbfgs_minimizer(**kwargs):

    def helper(fun, x0, args=(), **kwargs2):
        solver = LBFGS(fun=fun, **kwargs)
        res = solver.run(x0, *args)
        
        print("\n\n Single Basinhopping Iteration Complete \n\n")

        return OptimizeResult(
            x=np.array(res.params),
            fun=float(res.state.value),
            success=True,  # You can check res.state.error < tol for more rigor
            message="Local minimization finished via JAXOpt LBFGS",
            nit=int(res.state.iter_num)
        )
    return helper

optimizer_settings = {
    "maxiter": opt_maxiter,
    "tol": opt_tol,
    "verbose": True
}

method = create_lbfgs_minimizer(**optimizer_settings)

start_time = time.time()

solver = LBFGS(fun=objective, tol=opt_tol, maxiter=opt_maxiter, verbose=True)
result = solver.run(initial_guess, A_int, P_int, phi_theta_int)
result = result.params

# resultObj = basinhopping(
#     objective, 
#     initial_guess, 
#     niter=10, 
#     minimizer_kwargs={
#         "method": method, 
#         "args": (A_int, P_int, phi_theta_int)
#     }
# )
# result = resultObj.x
# resultVal = resultObj.fun

coeffs, u_right_interior = unpack(result, n)
coeffs = normalize_vec(coeffs)
end_time = time.time()
time5 = end_time - start_time

print(f"Time taken for optimization loop(s): {time5} seconds")
print("Part 4 Finished: Ran Optimization Loop")
print("\n\n --------------- \n\n")

'''
Part 5: Plot and Visualize Results
'''

# Get Even and Odd Modes
u_even,u_odd = femsystem.get_even_odd_modes(u_right_interior)
u_even_interior,u_odd_interior = u_even[femsystem.interior_dofs],u_odd[femsystem.interior_dofs]

energy = objective(result, A_int, P_int, phi_theta_int)
E_J, E_C, e0, lambda_y_val = ej_ec_e0(u_right_interior, A_int, P_int, phi_theta_int)

print(f"EJ: {E_J} | EC: {E_C} | e0: {e0}")
print(f"EJ/EC RATIO: {E_J/E_C}")

totalEndTime = time.time()
totalTime = totalEndTime - totalStartTime
print(f"Total Time Taken: {totalTime} seconds")

# Pickle the results
pickle_obj = {
    "parameters": {
        "N": N_val,
        "geometry": geometry,
        "separation": separation,
        "sidelenX": sidelenX,
        "sidelenY": sidelenY,
        "sidelen2X": sidelen2X,
        "sidelen2Y": sidelen2Y,
        "inner_dim": inner_dim,
        "padding": padding,
        "gridlenX": gridlenX,
        "gridlenY": gridlenY,
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
    "integrated_area": integrated_area,
    "coeffs": coeffs,
    "u_even": u_even,
    "u_odd": u_odd,
    "femsystem": femsystem,
    "totalTime": totalTime,
    "lambda_y_val": lambda_y_val
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


# Create the Hamiltonian Operator over Truncated Basis, to find eigenvectors for first excited state
def get_matrixel_and_qubit_freq(n,E_J,E_C,plot=False,coeffs=None):
    # Copy these three matrix construction functions from 3DTwoModesOpt.py
    x = (n-1)/2 - jnp.arange(n) # Charge Imbalance Eigenvalue, centered at 0

    def Jz2(n):
        j = (n-1)/2
        diagonals = j - jnp.arange(n)
        return jnp.diag(diagonals**2)

    def off_diag(n,k):
        ones_super, ones_sub = jnp.ones(n - k, dtype=jnp.int32),jnp.ones(n - k, dtype=jnp.int32)
        super_diag_matrix,sub_diag_matrix= jnp.diag(ones_super, k=k),jnp.diag(ones_sub, k=-1*k)
        result = super_diag_matrix + sub_diag_matrix
        return result

    def cos_phi(n):
        return off_diag(n,1) / 2

    hamiltonian = E_C * Jz2(n) - E_J * cos_phi(n) # No need for e_0 term, since we are only interested in eigenvectors

    # Find Eigenvalues and Eigenvectors
    eigenvalues,eigenvectors = jnp.linalg.eigh(hamiltonian)

    # Plot the ground state, first eigenvector
    ground_state_coeffs = eigenvectors[:,0] / jnp.sqrt(jnp.sum(eigenvectors[:,0]**2))
    first_excited_coeffs = eigenvectors[:,1] / jnp.sqrt(jnp.sum(eigenvectors[:,1]**2))

    if plot:
        ground_state_coeffs_from_optimization = coeffs / jnp.sqrt(jnp.sum(coeffs**2))
        plt.plot(x,ground_state_coeffs**2,".",color="red",label="Ground State (Solving Eigenvalue Problem)")
        plt.plot(x,ground_state_coeffs_from_optimization**2,"x",color="grey",label="Ground State (Optimization Loop)")
        plt.plot(x,first_excited_coeffs**2,".",color="green",label="First Excited State")
        plt.xlabel("Charge Imbalance Eigenvalue")
        plt.ylabel("Coefficient")
        plt.legend()
        plt.show()

    # Get the frequency of the qubit
    sorted_eigenvalues = jnp.sort(eigenvalues)
    omega_q = sorted_eigenvalues[1] - sorted_eigenvalues[0]

    return jnp.sum(first_excited_coeffs * ground_state_coeffs * x), omega_q # x is the charge imbalance eigenvalue

mat_el,E_q = get_matrixel_and_qubit_freq(n,E_J,E_C,plot=False,coeffs=coeffs)
print(f"Qubit Frequency: {E_q}")
