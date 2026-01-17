import numpy.typing as npt

import skfem as fem
from skfem.models.poisson import laplace
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import plot
from jax.experimental import sparse
from scipy.spatial import KDTree
from scipy.sparse.linalg import inv
import datetime

class FEMSystem:

    # Mesh, Element, Basis
    mesh:fem.Mesh = None 
    element:fem.AbstractBasis = None 
    intorder:int = 1
    basis:fem.Basis = None

    # Size of Problem
    elements:int = 0
    quad_per_element:int = 0
    dofs:int = 0

    # Boundary Information
    boundary_condition = 0
    all_dofs = None
    boundary_dofs = None
    interior_dofs = None

    # Weights
    weights = []

    # Interpolation Matrices
    phi_val = None
    phi_grad = None

    # Miscellanous
    dofmap = None
    node_coords_global = None
    coords_q_T = None
    X_ref = None
    W_ref = None
    flip_map = None

    saveFigsDir = None

    # Constructor - Preprocess Basis
    def __init__(self,mesh,element,intorder,boundary_condition=0,saveFigsDir=None):

        self.saveFigsDir = saveFigsDir

        # First set mesh, element, intorder and basis
        self.mesh = mesh
        self.element = element
        self.intorder = intorder
        self.basis = fem.CellBasis(mesh, element, intorder=intorder)

        # Step 1: Get boundary information
        self.boundary_condition = boundary_condition
        self.all_dofs = jnp.arange(self.basis.N)
        self.dofs = len(self.all_dofs)
        self.boundary_dofs = self.basis.get_dofs().flatten() # Empty call automatically gets boundary DOFs
        self.interior_dofs = jnp.setdiff1d(self.all_dofs, self.boundary_dofs)

        # Step 2: Get Weights
        weights = jnp.array(self.basis.dx) # Only for quadrature points, not necessarily the nodes
        elements,quad_per_element = weights.shape[0],weights.shape[1]
        self.weights,self.elements,self.quad_per_element = weights,elements,quad_per_element

        # Step 3: Get Interpolation Matrices, phi_val and phi_grad
        X_ref,W_ref = self.basis.quadrature
        n_local_dofs = element.doflocs.shape[0] # 3 for Triangle
        val_list = []
        grad_list = []
        # Loop over local nodes to get basis functions
        for i in range(n_local_dofs):
            dfield = element.gbasis(self.basis.mapping, X_ref, i)[0]
            val_list.append(dfield.value) # (elements,quadratures), value of ith basis function, at quadrature point, in this element
            grad_list.append(dfield.grad) # (dimensions,elements,quadratures), value of the derivative in a direction, of the ith basis function, at quadrature point, in this element
        phi_val = jnp.array(jnp.stack(val_list)).transpose(1, 2, 0) # eth index is interpolation matrix for element e
        phi_grad = jnp.array(jnp.stack(grad_list)).transpose(2, 1, 3, 0) #eth index, array at dth index, is interpolation matrix for element e 
        self.phi_val,self.phi_grad = phi_val,phi_grad

        # Step 4: Get Miscellanous Things
        self.dof_map = self.basis.element_dofs.T # (elements, dofs per element) matrix, maps to a global dof index        
        self.node_coords_global = jnp.array(mesh.doflocs.T)
        # self.dof_map = jnp.ascontiguousarray(self.basis.element_dofs.T.astype(jnp.int32))
        x_quad = self._interpolate_values(self.node_coords_global)
        self.coords_q_T = x_quad.transpose(2, 0, 1) # Cache This
        self.doflocs = self.basis.doflocs # arrays of x,y and z coordinates of the ith DOF
        self.X_ref,self.W_ref = X_ref,W_ref

        # Step 5: Get Flipping Mapping
        self.flip_map = self._generate_flip_mapping()

    def _save_fig(self,fig,plot_title):
        if not self.saveFigsDir: return
        if not plot_title: plot_title = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        fig.savefig(self.saveFigsDir+plot_title+".png")
    
    def _generate_flip_mapping(self,axis=0):
        coords = self.doflocs
        flipped_coords = self.doflocs.copy()
        flipped_coords[axis,:] *= -1

        # Do the Nearest Neighbor Search
        tree = KDTree(flipped_coords.T)

        distances, indices = tree.query(coords.T, k=1)

        return indices
    
    '''
    Arguments:
    - u_global: array of values at degrees of freedom. 
    '''
    def _interpolate_values(self,u_global):
        u_local_arr = u_global[self.dof_map] # for every element, get the actual dof value at the nodes in the element. Maps an array of length of total dofs to a (elements, dofs per element) matrix
        u_quad = jnp.einsum('eqd,ed... -> eq...',self.phi_val,u_local_arr) # for every element, the interpolated values of the quadrature points. Same dims as weights!!! e for "element", q for "quadrature", d for "degree of freedom / node"
        return u_quad

    def _interpolate_grad(self,u_global):
        u_local_arr = u_global[self.dof_map] # for every element, get the actual dof value at the nodes in the element
        grad_quad = jnp.einsum('exqd,ed -> xeq',self.phi_grad,u_local_arr) # add in axis "x", for the spatial dimension, direction to take gradient in. 
        return grad_quad
    
    def _interpolate_mat_interior(self,I):
        # 1. Create a Fast Lookup Table for Indices
        # Map: Global_ID -> Interior_ID (or -1 if boundary)
        # This acts like a hash map but is O(1) array access
        global_to_interior = jnp.full(self.dofs, -1, dtype=jnp.int32)
        global_to_interior = global_to_interior.at[I].set(jnp.arange(len(I)))

        # Construct Interpolation Matrix P (DOFs -> Quads)
        # We need a matrix where P[q, i] = phi_i(x_q)
        # femsystem.phi_val has shape (Elements, Quads, Local_DOFs)
        
        E = self.elements
        Q = self.quad_per_element
        L = self.element.doflocs.shape[0] # Local DOFs (e.g. 3 for triangles)
        N_dof = self.dofs
        Total_Quad = E * Q
        
        # Flatten phi_val to data array
        # phi_val_correct = jnp.transpose(self.phi_val, (0, 2, 1))
        data = jnp.array(self.phi_val).flatten() # Shape: (E * Q * Local_DOFs)
        
        # Create row indices (Quadrature points 0..Total_Quad)
        # Each quad has 3 local DOFs contributing to it
        rows = jnp.repeat(jnp.arange(Total_Quad), L)
        
        # Create col indices (Global DOF indices)
        # femsystem.dof_map has shape (E, 3) -> we need to broadcast to (E, Q, 3)
        # because the same 3 global DOFs apply to all Q quads in that element
        cols = jnp.broadcast_to(self.dof_map[:, jnp.newaxis, :], (E, Q, L)).flatten()

        cols_interior = global_to_interior[cols]
        mask = cols_interior != -1
        # Apply mask to everything
        rows_final = rows[mask]
        cols_final = cols_interior[mask]
        data_final = data[mask]

        indices = jnp.stack([rows_final, cols_final], axis=1)
        # Build Sparse P
        P = sparse.BCOO((data_final, indices), shape=(Total_Quad, len(I)))
        
        return P
    
    def _complete_arr(self,interior_vals):
        u_full = jnp.zeros(self.dofs)
        u_full = u_full.at[self.interior_dofs].set(interior_vals)
        return u_full
    
    def _get_at_interior_dofs(self,func):
       # For each row, get only interior DOFs. 2D will have two rows for x,y, 3D will have 3 rows for x,y,z
        filtered_doflocs = self.doflocs[:,self.interior_dofs]

        # Pass each row as an argument, by "*"
        interior_vals = func(*filtered_doflocs) 
        return interior_vals

    # With boundary conditions
    def _get_at_dofs(self,func):
        interior_vals = self._get_at_interior_dofs(func)
        full_vals = self._complete_arr(interior_vals) # values at nodes 
        return full_vals
    
    def _get_u_from_interior(self,u_interior):
        u_final = jnp.ones((self.dofs)) * self.boundary_condition
        # Use .at[].set() for functional update, this is the "JAX" way
        u_final = u_final.at[self.interior_dofs].set(u_interior)
        return u_final
    
    def _plot_u_2d(self,u,plot_title):
        ax = plot(self.basis, u, shading='gouraud')

        if ax.collections:
            plt.colorbar(ax.collections[0])
        
        plt.colorbar(ax.collections[0])
        plt.title(plot_title)

        # Save Fig
        fig = ax.get_figure()
        self._save_fig(fig,plot_title)

        plt.show()
        


    def _plot_u_2d_in_3d(self,u,plot_title):
        x_nodes,y_nodes = self.doflocs
        triangles = self.basis.mesh.t.T
        z_values = u

        # 2. Create 3D Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_trisurf(x_nodes, y_nodes, z_values, 
                            triangles=triangles, 
                            cmap='viridis', 
                            edgecolor='none',
                            linewidth=0,
                            antialiased=False)
        # 3. Add labels and colorbar
        ax.set_title(plot_title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('u(x,y)')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


        # Save Fig
        fig = ax.get_figure()
        self._save_fig(fig,plot_title)

        plt.show() 
    
    '''
    Arguments:
    - vals: values at quadratures, flattened array.
    - plot_title: plot title
    '''
    def plot_at_quad_2d(self,vals,plot_title=""):
        coords = self.basis.mapping.F(self.X_ref) 
        flat_coords = coords.reshape(2, -1)

        plt.figure(figsize=(8, 8))
        sc = plt.scatter(flat_coords[0], flat_coords[1], c=vals, s=5, cmap='viridis')

        # 3. Add colorbar and formatting
        plt.colorbar(sc)
        plt.title(plot_title)
        plt.axis('equal')

        # Save Fig
        self._save_fig(plt.gcf(),plot_title)

        plt.show()
    
    def _plot_3d(self,coords,vals,plot_title=""):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the data
        sc = ax.scatter(coords[0], coords[1], coords[2], c=vals, s=5, cmap='viridis')
        
        # Add colorbar and formatting
        plt.colorbar(sc)
        plt.title(plot_title)
        
        # Save Fig
        self._save_fig(plt.gcf(),plot_title)
        
        plt.show()

        return fig, ax, sc
    
    def plot_3d_interior(self,u_interior,plot_title=""):
        u = self._get_u_from_interior(u_interior)
        coords = self.node_coords_global.T

        self._plot_3d(coords,u,plot_title)

    def plot_at_quad_3d(self,vals,plot_title=""):

        coords = self.basis.mapping.F(self.X_ref) 
        flat_coords = coords.reshape(3, -1)

        self._plot_3d(flat_coords,vals,plot_title)

    def plot_at_quad_3d_sliced(self, vals, plot_title="", slice_axis='z', slice_val=0.5, tol=0.05):
        coords = self.basis.mapping.F(self.X_ref) 
        flat_coords = coords.reshape(3, -1)
        flat_vals = vals.flatten()
        
        x, y, z = flat_coords[0], flat_coords[1], flat_coords[2]
        
        # Filter points based on slice
        if slice_axis == 'z':
            mask = jnp.abs(z - slice_val) < tol
        elif slice_axis == 'y':
            mask = jnp.abs(y - slice_val) < tol
        else: # x
            mask = jnp.abs(x - slice_val) < tol
            
        # Apply mask
        xs, ys, zs = x[mask], y[mask], z[mask]
        vs = flat_vals[mask]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(xs, ys, zs, c=vs, s=10, cmap='viridis', alpha=0.8)

        plt.colorbar(sc)
        plt.title(f"{plot_title} (Slice @ {slice_axis}={slice_val:.3f})")
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max()); ax.set_zlim(z.min(), z.max())
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        # Save Fig
        self._save_fig(plt.gcf(),plot_title)
        
        plt.show()
    
    def plot_n_slices_interior(self,u_interior,n_slices=5,slice_axis="z",plot_title="",tol=None):
        coords = self.node_coords_global.T
        x, y, z = coords[0], coords[1], coords[2]
        if slice_axis == 'x':
            axis_vals = x
        elif slice_axis == 'y':
            axis_vals = y
        else:  # 'z'
            axis_vals = z

        half_dist = (axis_vals.max() - axis_vals.min()) / (2*n_slices)
        if tol is None:
            tol = half_dist
        
        for i in range(n_slices):
            slice_val = i * (1 / n_slices) * (axis_vals.max() - axis_vals.min()) + axis_vals.min() + half_dist
            self.plot_interior_at_quad_3d_sliced(u_interior,slice_val,slice_axis,plot_title,tol)

    '''
    Arguments:
    - func(x,y): function to plot
    '''
    def plot_func_2d(self,func,plot_title=""):
        u_final = self._get_at_dofs(func)
        self._plot_u_2d(u_final,plot_title)
    
    def plot_func_2d_in3d(self,func,plot_title=""):
        u_final = self._get_at_dofs(func)
        self._plot_u_2d_in_3d(u_final,plot_title)


    '''
    Arguments: 
    - u_interior: array of interior points
    - plot_title: plot title
    '''
    def plot_at_interior_2d(self,u_interior,plot_title=""):
        u_final = self._get_u_from_interior(u_interior)
        self._plot_u_2d(u_final,plot_title)
    
    def plot_at_interior_2d_in3d(self,u_interior,plot_title=""):
        u_final = self._get_u_from_interior(u_interior)
        self._plot_u_2d_in_3d(u_final,plot_title)

    def plot_interior_at_quad_3d(self,u_interior,plot_title="Values at Quadratures, for a 3D Function"):
        u_global = self._get_u_from_interior(u_interior)
        u_quad = self._interpolate_values(u_global)
        self.plot_at_quad_3d(u_quad,plot_title)  
    
    def plot_interior_at_quad_3d_sliced(self,u_interior,slice_val,slice_axis="z",plot_title="Values at Quadratures, for a 3D Function Slice",tol=0.05):
        u_global = self._get_u_from_interior(u_interior)
        u_quad = self._interpolate_values(u_global)

        self.plot_at_quad_3d_sliced(u_quad,plot_title=plot_title,slice_axis=slice_axis,slice_val=slice_val,tol=tol)

        
    '''
    Arguments: 
    - func(x,y): function to plot, with boundary conditions at self.boundary_condition
    - plot_title: plot title
    '''
    def plot_values_2d(self,func,plot_title="Values at Quadratures, for a 2D Function"):
        u_global = self._get_at_dofs(func)

        # Get at quadratures and plot
        u_quad = self._interpolate_values(u_global)
        self.plot_at_quad_2d(u_quad,plot_title)

    def plot_grad_squared_2d(self,func,plot_title="Grad Squared at Quadratures, for a 2D Function"):
        u_global = self._get_at_dofs(func)

        # Get at quadratures and plot
        grad_quadx,grad_quady = self._interpolate_grad(u_global)
        laplacian_quad = grad_quadx**2 + grad_quady**2
        self.plot_at_quad_2d(laplacian_quad,plot_title)

    
    def plot_values_3d(self,func,plot_title="Values at Quadratures, for a 3D Function"):
        u_global = self._get_at_dofs(func)

        # Get at quadratures and plot
        u_quad = self._interpolate_values(u_global)
        self.plot_at_quad_3d(u_quad,plot_title) 
    
    '''
    Arguments:
    - func(x,y,z): function to plot, with boundary conditions at self.boundary_condition
    - slice_axis: "x", "y" or "z"
    - slice_val: the value along the axis to take the slice
    - tol: tolerance around the slice_val to be plotted
    '''
    def plot_values_3d_sliced(self,func,slice_val,slice_axis="z",plot_title="Values at Quadratures, for a 3D Function Slice",tol=0.05):
        u_global = self._get_at_dofs(func)

        # Get at quadratures and plot
        u_quad = self._interpolate_values(u_global)
        self.plot_at_quad_3d_sliced(u_quad,plot_title=plot_title,slice_axis=slice_axis,slice_val=slice_val,tol=tol)

    '''
    Arguments:
    - func(x_vec): analytical function to compare to
    - u_global: solved for u_global to compare to analytical function
    '''
    def compare_at_quads(self,func,u_global):
        u_quad = self._interpolate_values(u_global)
 
        f_quad = func(self.coords_q_T)

        diff_sq = (u_quad - f_quad) ** 2
        return diff_sq
        
    '''
    Arguments:
    - func(u,grad_u,x): where grad_u and x are multidimensional vectors. MUST return a scalar
    - u_global: array of u at degrees of freedom
    '''
    def integrate(self,func,u_global):
        u_quad = self._interpolate_values(u_global)
        grad_quad = self._interpolate_grad(u_global)

        L_density = func(u_quad, grad_quad, self.coords_q_T)
        integral_result = jnp.sum(L_density * self.weights)
        return integral_result

    '''
    Arguments:
    - func(u1,grad_u1,u2,grad_u2,x): where grad_u1/2 and x are multidimensional vectors. MUST return a scalar
    - u1_global: array of u1 at degrees of freedom
    - u2_global: array of u2 at degrees of freedom
    '''
    def integrate_two(self,func,u1_global,u2_global):
        u1_quad = self._interpolate_values(u1_global)
        grad1_quad = self._interpolate_grad(u1_global)
        u2_quad = self._interpolate_values(u2_global)
        grad2_quad = self._interpolate_grad(u2_global)

        L_density = func(u1_quad,grad1_quad,u2_quad,grad2_quad,self.coords_q_T)
        integral_result = jnp.sum(L_density * self.weights)
        return integral_result
    
    '''
    Arguments:
    - kernel_func(x_vec,y_vec): returns scalar, interaction function
    - tol: tolerance, if absolute value is below tol, just set to zero. Returns a JAX spare matrix
    '''  
    def get_sparse_interaction_mat(self, kernel_func, tol=1e-5):
        # (Dim, E, Q) -> (Total_Points, Dim)
        coords_flat = self.coords_q_T.transpose(1, 2, 0).reshape(-1, self.coords_q_T.shape[0])
        
        # Expand dimensions to broadcast: (N, 1, D) vs (1, N, D) -> (N, N, D)
        x_in = coords_flat[:, None, :]
        y_in = coords_flat[None, :, :]
        
        # Compute Dense Kernel
        K_dense = kernel_func(x_in, y_in)
        
        # Apply Cutoff
        K_dense = jnp.where(jnp.abs(K_dense) > tol, K_dense, 0.0)
        
        # Convert to Sparse
        K_sparse = sparse.BCOO.fromdense(K_dense)
        return K_sparse
    
    def _double_integral_preprocess(self,func1,func2,u1_global,u2_global):
        u1_quad = self._interpolate_values(u1_global)
        grad1_quad = self._interpolate_grad(u1_global)
        u2_quad = self._interpolate_values(u2_global)
        grad2_quad = self._interpolate_grad(u2_global) 

        # Evaluate at quadratures, should each return a (elements, quadratures per element matrix), of scalars, representing evaluated scalar value at the quadrature point
        func_1_eval = func1(u1_quad,grad1_quad,u2_quad,grad2_quad,self.coords_q_T)
        func_2_eval = func2(u1_quad,grad1_quad,u2_quad,grad2_quad,self.coords_q_T)
        weighted_f1 = func_1_eval * self.weights # Multiply by weights now, for convenience. 
        weighted_f2 = func_2_eval * self.weights

        # Flatten, so just an array of values at quadrature points:
        weighted_f1_flat = weighted_f1.ravel() # ravel flattens in a way consistent with ordering of quadratures
        weighted_f2_flat = weighted_f2.ravel()
        return weighted_f1_flat,weighted_f2_flat
    
    '''
    Arguments:
    - func1(u1,grad_u1,u2,grad_u2,x): where grad_u1/2 and x are multidimensional vectors. MUST return a scalar
    - func2(u1,grad_u1,u2,grad_u2,y): where grad_u1/2 and y are multidimensional vectors. MUST return a scalar
    - interaction_matrix: quadratures x quadratures matrix for interactions
    - u2_global: array of u2 at degrees of freedom
    ''' 
    def double_integral(self,func1,func2,interaction,P_int,u1_global,u2_global):
        weighted_f1_flat,weighted_f2_flat = self._double_integral_preprocess(func1,func2,u1_global,u2_global)

        v1 = P_int.T @ weighted_f1_flat
        v2 = P_int.T @ weighted_f2_flat
        res = v1 @ interaction @ v2

        return res

    '''
    Writing in Progress, only hypothetically needed for extremely large meshes
    ''' 
    # Optimized, using just Bilinear form A (inverse of kernel matrix).
    def double_integral2(self,func1,func2,bilinear_form,u1_global ,u2_global):
        weighted_f1_flat,weighted_f2_flat = self._double_integral_preprocess(func1,func2,u1_global,u2_global)

        # First we need to only take interior
        weighted_f1_flat_interior,weighted_f2_flat_interior = weighted_f1_flat[self.interior_dofs], weighted_f2_flat[self.interior_dofs]

        # Compute G @ weighted_f2_flat by the Conjugate Gradient Method



    def greens(self,dof_source,dof_response):
        u_global = jnp.zeros(len(self.all_dofs))
        u_global = u_global.at[dof_source].set(1)

        loc_response = self.doflocs[:dof_response] # location x,y,z of response
        func = lambda u,grad_u,x: u * 1/ (2*jnp.pi) * jnp.log(jnp.linalg.norm(x - loc_response))

        return self.integrate(func,u_global)
    
    '''
    Arguments:
    - func(u,grad_u,x): where grad_u and x are multidimensional vectors. MUST return a scalar
    - u_global: matrix of (n,dofs). Each row is a set of u's at the degrees of freedom
    '''
    # def vec_integrate(self,func,u_global):
    #     u_quad = self._interpolate_values(u_global)
    #     grad_quad = self._interpolate_grad(u_global)
    #     x_quad = self._interpolate_values(self.node_coords_global) # coordinates of quadrature points

    #     coords_q_T = x_quad.transpose(2, 0, 1)


    #     L_density = func(u_quad, grad_quad, coords_q_T)



    #     integral_result = jnp.sum(L_density * self.weights)
    #     return integral_result 

    ''' Only intended for test purposes
    Arguments:
    - func(x): function of x (array of dimension d) you want to integrate
    '''
    def integrate_function(self,func):
        integrand_quad = func(self.coords_q_T)
        integral_result = jnp.sum(integrand_quad * self.weights)
        return integral_result
    
    def get_greens_kernel(self):

        # At DOFs 
        A = fem.asm(laplace,self.basis)
        A_int, xI, I = fem.condense(A, D=self.boundary_dofs, expand=True)
        A_inv_int = jnp.array(inv(A_int).toarray())

        P_int = self._interpolate_mat_interior(I)

        return A_inv_int, P_int
        
    '''
    Arguments:
    - u_interior: array of values only at interior dofs
    - objective(u_global): your objective function. Passes in full normalized u_global
    '''
    def apply_bc_and_norm(self,u_interior):
        u_full = jnp.ones(self.dofs) * self.boundary_condition
        u_full = u_full.at[self.interior_dofs].set(u_interior)
        u_norm = self.integrate(lambda u,a,b: u**2,u_full)

        # Preventing Divide by Zero
        safe_norm = jnp.where(u_norm > 0, u_norm, 1.0)
        u_full = jnp.where(u_norm > 0, u_full / jnp.sqrt(safe_norm),u_full)

        return u_full
    
    def apply_bc(self,u_interior):
        u_full = jnp.ones(self.dofs) * self.boundary_condition
        u_full = u_full.at[self.interior_dofs].set(u_interior)
        return u_full
    
    def separate_even_odd_apply_by_and_norm(self,u_interior):
        u_full = jnp.ones(self.dofs) * self.boundary_condition
        u_full = u_full.at[self.interior_dofs].set(u_interior) 
        u_full_flipped = u_full[self.flip_map] # Literally just reindexes

        u_even = u_full + u_full_flipped
        u_odd = u_full - u_full_flipped

        u_even_norm = self.integrate(lambda u,a,b: u**2,u_even)
        u_odd_norm = self.integrate(lambda u,a,b: u**2,u_odd)

        safe_even_norm = jnp.where(u_even_norm > 0, u_even_norm, 1.0)
        safe_odd_norm = jnp.where(u_odd_norm > 0, u_odd_norm, 1.0)

        u_even = jnp.where(u_even_norm > 0, u_even / jnp.sqrt(safe_even_norm),u_even)
        u_odd = jnp.where(u_odd_norm > 0, u_odd / jnp.sqrt(safe_odd_norm),u_odd)
        
        return u_even,u_odd
    
    def get_initial_ones_interior(self):
        return jnp.ones(len(self.interior_dofs))
    
    def get_initial_random_interior(self):
        key = jax.random.PRNGKey(0)
        random_array_uniform = jax.random.uniform(key, shape=(len(self.interior_dofs)), minval=0.0, maxval=1.0)
        return random_array_uniform 
    
    def ones_on_island(self,theta_func):
        return jnp.where(theta_func(self.doflocs),1.0,0.0)[self.interior_dofs]
