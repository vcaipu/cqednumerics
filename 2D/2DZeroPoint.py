# Import the FEMSystem Class from directory above
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
import jax
import matplotlib.pyplot as plt
import argparse
import pickle


'''
Handle Command Line Args
'''

parser = argparse.ArgumentParser(description="")
parser.add_argument("--plotdir", type=str, help="Directory to save all plots. MUST end with a slash /")
parser.add_argument("--sidelen", type=float, help="Sidelength of island. Default set to 20",default=20.0)
parser.add_argument("--separation", type=float, help="Gap between islands. Default set to 20",default=20.0)
parser.add_argument("--N", type=int, help="Total number of particles. Default is 70",default=70)
parser.add_argument("--n", type=int, help="Max number difference to be considered, in computational domain. Default is 100",default=100)
args = parser.parse_args()
plotdir = args.plotdir
sidelen = args.sidelen
separation = args.separation
n = args.n # Number of coefficients. NOTE: Just set this to an outside variable. Lots of trouble trying to pass into a dynamical argument, since JAX doesn't like when array indices are dynamical. 
N = args.N #Total number of particles

print(f"RUNNING WITH SEPARATION {separation}")

# Make the plotdirs directory
os.makedirs(plotdir, exist_ok=True)


'''
Part 1: Create Mesh
'''

print("Starting Part 1: Creating the Mesh")
# Create the FEMSystem Object
mesh = fem.MeshTri.init_sqsymmetric().refined(4)
L = 60.0
mesh = mesh.scaled(2 * L).translated((-L, -L))
element = fem.ElementTriP1()
intorder = 4

femsystem = FEMSystem(mesh,element,intorder,boundary_condition=0,saveFigsDir=plotdir)
print("Part 1 Finished: Mesh Created")
print(f"Degrees of Freedom: {femsystem.dofs}")
print("\n\n --------------- \n\n")

'''
Part 2: Define Geometry
'''

print("Starting Part 2: Defining the Geometry in 2D")

sideLen = sidelen 
centerLeft,centerRight = ((sideLen + separation)/2,0), (-(sideLen + separation)/2,0)
area = 2 * (sideLen ** 2)

def theta(x_vec):
    x,y = x_vec[0],x_vec[1]
    cond1 = (jnp.abs(x-centerLeft[0]) <= sideLen / 2) & (jnp.abs(y-centerLeft[1]) <= sideLen / 2)
    cond2 = (jnp.abs(x-centerRight[0]) <= sideLen / 2) & (jnp.abs(y-centerRight[1]) <= sideLen / 2)
    return cond1 | cond2

def theta_right_only(x_vec):
    x,y = x_vec[0],x_vec[1]
    cond1 = (jnp.abs(x-centerLeft[0]) <= sideLen / 2) & (jnp.abs(y-centerLeft[1]) <= sideLen / 2)
    cond2 = (jnp.abs(x-centerRight[0]) <= sideLen / 2) & (jnp.abs(y-centerRight[1]) <= sideLen / 2)
    return cond1

# Now get integrated area. 
theta_at_dofs = theta(femsystem.doflocs).astype(jnp.float32)
integrated_area = femsystem.integrate(lambda u,grad_u,x: u,theta_at_dofs)

print(f"Area Computed: {area} | Area Integrated: {integrated_area}")
print(f"Plotting Geometry...")
femsystem.plot_func_2d(lambda a,b: theta([a,b]),plot_title="Part 2: Geometry")

print("Part 2 Finished: Defined Geometry")
print("\n\n --------------- \n\n")

'''
Part 3: Define Objective Function
'''

print("Starting Part 3: Defining Objective Function")

'''
Helper Functions for Integrals
'''

def laplacian(u,grad_u,x):
    return -1*jnp.sum(grad_u**2,axis=0)

# For Potential Energy Double Inetgral, u1 is our function phi_{+/-} and u2 is theta. 
def u_squared(u,grad_u,u2,grad_u2,x):
    return u**2

# Define Pre-Computed Values for Theta and Green's Function:
def theta_func(u,grad_u,u2,grad_u2,x):
    return u2

# U_{++++} or U_{----}, Really N * \alpha
def alpha(u,G_mat,P_int):
    return integrated_area  * femsystem.double_integral(lambda u1,a,b,c,d: u1**2,lambda u1,a,b,c,d: u1**2,G_mat,P_int,u,u)

# U_{+--+} = U{-++-} - Remember middle two are wrt to y, Outer two wrt to x, from notation used in doc
def beta(u1_arg,u2_arg,G_mat,P_int):
    return integrated_area * femsystem.double_integral(lambda u1,a,b,c,d: u1**2, lambda a,b,u2,c,d: u2**2, G_mat, P_int,u1_arg,u2_arg)

# U_{++--} = U{+-+-}
def gamma(u1_arg,u2_arg,G_mat,P_int):
    return integrated_area * femsystem.double_integral(lambda u1,a,u2,c,d: u1*u2, lambda u1,b,u2,c,d: u1*u2, G_mat, P_int,u1_arg,u2_arg)

'''
Helper Functions for Matrices
'''

# N x N, with k off diagonal all 1s
def off_diag(N,k):
    ones_super, ones_sub = jnp.ones(N - k, dtype=jnp.int32),jnp.ones(N - k, dtype=jnp.int32)
    super_diag_matrix,sub_diag_matrix= jnp.diag(ones_super, k=k),jnp.diag(ones_sub, k=-1*k)
    result = super_diag_matrix + sub_diag_matrix
    return result 

def cos_phi(N):
    return off_diag(N,1) / 2

def cos_2phi(N):
    return off_diag(N,2) / 2

def Jz(N):
    j = (N-1)/2
    diagonals = j - jnp.arange(N)
    return jnp.diag(diagonals)

def Jz2(N):
    j = (N-1)/2
    diagonals = j - jnp.arange(N)
    return jnp.diag(diagonals**2)

def expval(mat,vec):
    return jnp.vdot(vec,mat @ vec)

def normalize_vec(vec):
    norm_v = jnp.linalg.norm(vec)
    normalized_v = jnp.where(jnp.isclose(norm_v, 0.0), vec, vec / norm_v )
    return normalized_v

def guess_gaussian(N,stddevs=4):
    x = jnp.linspace(-stddevs, stddevs, N)
    mu,sigma = 0.0,1.0
    exponent = -jnp.square(x - mu) / (2.0 * jnp.square(sigma))
    gaussian_array = jnp.exp(exponent)
    return gaussian_array

# get first N as the vector of coeffs, remaining as u_interior
def unpack(vec,N):
    coeff_vec,u = vec[:N],vec[N:]
    return coeff_vec,u

# VERY VERY IMPORTANT TO PASS IN G_mat AS AN ARGUMENT, AND SET TO CONSTANT IN OPTIMIZATION LOOP
# This is because when JAX compiles this function, it will treat the G_mat as a "tracer", so just any matrix of constants with some shape. 
# If you hardcode it into the function, it will treat it as an actual part of the code and will spent time compiling a massive amount of hardcoded values as "code" essentially. This is why it takes almost 10 minutes to run first optimization iteration. 
# @jax.jit

'''
Term 1: Zero Point Energy
'''
def epsilon_func(u_global,G_mat,P_int,theta_at_dofs):
    # Kinetic Term
    kinetic = -4 * femsystem.integrate(laplacian,u_global)

    # Potential Term
    potential = -2 * femsystem.double_integral(u_squared,theta_func,G_mat,P_int,u_global,theta_at_dofs)

    return kinetic  + potential

def E(u_global,G_mat,P_int,theta_at_dofs):
    return epsilon_func(u_global,G_mat,P_int,theta_at_dofs) + alpha(u_global,G_mat,P_int)

def zeropoint(u_even,u_odd,G_mat,P_int,theta_at_dofs):
    E_plus,E_minus = E(u_even,G_mat,P_int,theta_at_dofs), E(u_odd,G_mat,P_int,theta_at_dofs)
    return (E_plus + E_minus) / 2

'''
Term 2: S_z h_z
'''

def h_z(u_even,u_odd,G_mat,P_int,theta_at_dofs):
    return epsilon_func(u_even,G_mat,P_int,theta_at_dofs) - epsilon_func(u_odd,G_mat,P_int,theta_at_dofs) + alpha(u_even,G_mat,P_int) - alpha(u_odd,G_mat,P_int)

'''
Before you start the optimization loop:
1. Define Objective
2. Compute Interaction Kernel
3. Get Initial Guess
'''


# 1. Defining Objective
@jax.jit
def objective(vec,G_mat,P_int,theta_at_dofs):
    # Unpack the modes from the coefficients
    coeff_vec,u_interior = unpack(vec,n)

    # Normalize Coeff Vector: 
    coeff_vec_norm = normalize_vec(coeff_vec)

    # Unpack even and odd modes
    u_even,u_odd = femsystem.separate_even_odd_apply_by_and_norm(u_interior)
    E_plus,E_minus = E(u_even,G_mat,P_int,theta_at_dofs), E(u_odd,G_mat,P_int,theta_at_dofs)

    # Construct Objective
    e0 = ( E_plus + E_minus ) / 2 - gamma(u_even,u_odd,G_mat,P_int) / N
    hz = ( E_plus - E_minus ) 

    cos1= cos_phi(n) #,cos_2phi(N)
    first_harmonic = expval(cos1,coeff_vec_norm) * hz / 2

    lambda_x = 4*gamma(u_even,u_odd,G_mat,P_int)
    jz2 = Jz2(n)
    capacitive = lambda_x / (N**2) * expval(jz2,coeff_vec_norm)

    return e0 + capacitive + first_harmonic

# 2. Computing Interaction Kernel
G_mat,P_int = femsystem.get_greens_kernel()

# 3. Getting Initial Guess
coeff_vector_init = guess_gaussian(n,10)
normed = normalize_vec(coeff_vector_init)
print(jnp.sum(coeff_vector_init**2),jnp.sum(normed**2))
print("Gaussian First Guess")
u_interior_init = femsystem.ones_on_island(theta_right_only)
initial_guess = jnp.concatenate((coeff_vector_init, u_interior_init), axis=0)

'''
Testing, for a sanity check, and to do a jit compilation
'''
temp = objective(initial_guess,G_mat,P_int,theta_at_dofs)

print(f"Plotting the Initial Guess")
femsystem.plot_at_interior_2d_in3d(u_interior_init,plot_title="Initial Guess")
print("Part 3 Finished: Defined Objective Function")
print("\n\n --------------- \n\n")

'''
Part 4: Run Optimizaton Loop
'''

print("Starting Part 4: Running Optimization Loop")
solver = LBFGS(fun=objective,tol=1e-6,verbose=True)
result = solver.run(initial_guess,G_mat,P_int,theta_at_dofs)
result = result.params 
coeffs,u_interior = unpack(result,n)
print("Part 4 Finished: Ran Optimization Loop")
print("\n\n --------------- \n\n")

'''
Part 5: Save Plots
'''
print("Starting Part 5: Saving Plots")

u_even,u_odd = femsystem.separate_even_odd_apply_by_and_norm(u_interior)
u_even_interior,u_odd_interior = u_even[femsystem.interior_dofs],u_odd[femsystem.interior_dofs]

femsystem.plot_at_interior_2d_in3d(u_odd_interior,plot_title="Odd Mode")
femsystem.plot_at_interior_2d_in3d(u_even_interior,plot_title="Even Mode")

energy = objective(result,G_mat,P_int,theta_at_dofs)

x = (n-1)/2 - jnp.arange(n)
fig, ax = plt.subplots(figsize=(8, 6)) # Creates a figure and a single subplot (axes)
ax.plot(x,coeffs,".")
ax.set_xlabel('Charge Imbalance Eigenvalue')
ax.set_ylabel('Coefficient Value')
femsystem._save_fig(plt.gcf(),"Coefficients")

# Pickle the results
pickle_obj = {
    "n": n,
    "N": N,
    "objective": energy, # Final objective value
    "theta_at_dofs": theta_at_dofs,
    "coeffs": coeffs,
    "u_even": u_even,
    "u_odd": u_odd,
    "femsystem": femsystem
}
with open(plotdir+"results.pkl", 'wb') as f:
    pickle.dump(pickle_obj,f)

print("Part 5 Finished: Saving Plots")


lambda_x = 4*gamma(u_even,u_odd,G_mat,P_int)
print(f"Separation = {separation} | Lambda_x: {lambda_x}")