from gmshgen3d import generate_mesh # Before changing directories.
# First set to no constant folding 
import os
import sys
import skfem as fem
os.environ["JAX_NO_CONSTANT_FOLD"] = "true"


# Import the FEMSystem Class from directory above
# Import the FEMSystem Class from directory above
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from FEMSystem import FEMSystem

import jax.numpy as jnp

seps = [1,5,10,20,30,50]
cross_integrals = []

for sep in seps:
    mesh_file_path = "./custommeshdebug.msh"

    sidelen = 10.0
    padding = 20.0
    gridlenX = sidelen + sep + 2 * padding
    gridlenY = sidelen + 2 * padding
    gridlenZ = sidelen + 2 * padding
    inner_dimX,inner_dimY,inner_dimZ = sidelen/2,sidelen/2,sidelen/2
    lc_large,lc_small = 3,0.2

    generate_mesh(gridlens=(gridlenX, gridlenY, gridlenZ), sidelens=(sidelen, sidelen, sidelen), inner_dims=(inner_dimX, inner_dimY, inner_dimZ), separation=sep, lc_large=lc_large, lc_small=lc_small, output_file=mesh_file_path)

    # Construct Theta (the geometry function), and take the double integral of it with the Green's Kernel
    mesh = fem.Mesh.load(mesh_file_path)

    element = fem.ElementTetP1()
    intorder = 4
    femsystem = FEMSystem(mesh, element, intorder, boundary_condition=0)
    print("Done Constructing FEMSystem")

    centerLeft,centerRight = ((sidelen+sep)/2,0,0), (-(sidelen+sep)/2,0,0)

    def theta(x_vec):
        x,y,z = x_vec[0],x_vec[1],x_vec[2]
        cond1 = (jnp.abs(x-centerLeft[0]) <= sidelen / 2) & (jnp.abs(y-centerLeft[1]) <= sidelen / 2) & (jnp.abs(z-centerLeft[2]) <= sidelen / 2)
        cond2 = (jnp.abs(x-centerRight[0]) <= sidelen / 2) & (jnp.abs(y-centerRight[1]) <= sidelen / 2) & (jnp.abs(z-centerRight[2]) <= sidelen / 2)
        return cond1 | cond2

    def theta_left(x_vec):
        x,y,z= x_vec[0],x_vec[1],x_vec[2]
        cond1 = (jnp.abs(x-centerLeft[0]) <= sidelen / 2) & (jnp.abs(y-centerLeft[1]) <= sidelen / 2) & (jnp.abs(z-centerLeft[2]) <= sidelen / 2)
        return cond1

    def theta_right(x_vec):
        x,y,z = x_vec[0],x_vec[1],x_vec[2]
        cond1 = (jnp.abs(x-centerRight[0]) <= sidelen / 2) & (jnp.abs(y-centerRight[1]) <= sidelen / 2) & (jnp.abs(z-centerRight[2]) <= sidelen / 2)
        return cond1

    theta_at_dofs = theta(femsystem.doflocs).astype(jnp.float32)
    theta_left_at_dofs = theta_left(femsystem.doflocs).astype(jnp.float32)
    theta_right_at_dofs = theta_right(femsystem.doflocs).astype(jnp.float32)

    integrated_volume = femsystem.integrate(lambda u,grad_u,x: u,theta_at_dofs)
    volume = (10**3) * 2
    print(f"Area: {volume} | Integrated Area Estimate: {integrated_volume}")


    print("Part 2 Finished: Defined Geometry")
    print("\n\n --------------- \n\n")


    A_int,P_int = femsystem.get_stiffness_matrix()

    def first_func(u,grad_u,u2,grad_u2,x):
        return u
        
    def second_func(u,grad_u,u2,grad_u2,x):
        return u2

    def potential_integral(u_1,u_2):
        u1_quad = femsystem._interpolate_values(u_1)
        weighted_u1 = u1_quad * femsystem.weights
        v = P_int.T @ weighted_u1.ravel()
        x = femsystem._linear_solve_cg(A_int, v)

        u2_quad = femsystem._interpolate_values(u_2)
        weighted_u2 = (u2_quad) * femsystem.weights
        v_u2 = P_int.T @ weighted_u2.ravel()
        potential = (v_u2 @ x)
        return potential

    theta_pos = theta_left_at_dofs + theta_right_at_dofs
    theta_neg = theta_right_at_dofs - theta_left_at_dofs

    total_integral = (potential_integral(theta_pos,theta_pos) - potential_integral(theta_neg,theta_neg)) / 4
    left_integral = potential_integral(theta_left_at_dofs,theta_right_at_dofs)
    right_integral = potential_integral(theta_right_at_dofs,theta_left_at_dofs)
    cross_integral = femsystem.double_integral_cg(first_func,second_func,A_int,P_int,theta_left_at_dofs,theta_right_at_dofs)


    self_interactions = potential_integral(theta_left_at_dofs,theta_left_at_dofs) + potential_integral(theta_right_at_dofs,theta_right_at_dofs)
    cross_interactions = 2*potential_integral(theta_left_at_dofs,theta_right_at_dofs)
    ec_integral = potential_integral(theta_neg,theta_neg)

    print(f"Cross Integral: {cross_integral}")
    print(f"Left Integral: {left_integral}")
    print(f"Right Integral: {right_integral}")
    print(f"Total Integral: {total_integral}")
    print("\n\n")
    print(f"EC Integral: {ec_integral}")
    print(f"Self Interactions: {self_interactions}")
    print(f"Cross Interactions: {cross_interactions}")
    print(f"Check Consistency: {self_interactions - cross_interactions == ec_integral}")

    cross_integrals.append([cross_integral,left_integral,right_integral,ec_integral,self_interactions,cross_interactions])

print("Separation | Cross Integral\n\n\n")
print(seps)
for sep, integrals in zip(seps, cross_integrals):
    print(f"{sep} | {integrals[0]} | {integrals[1]} | {integrals[2]} | {integrals[3]} | {integrals[4]} | {integrals[5]}")