class FEMSystem:
    
    def __init__():
        weights = jnp.array(basis.dx) # Only for quadrature points, not necessarily the nodes
        boundary_dofs = basis.get_dofs().flatten()
        elements,quad_per_element = weights.shape[0],weights.shape[1]

        X_ref = basis.quadrature[0]

        n_local_dofs = element.refdom.nnodes # 3 for Triangle
        val_list = []
        grad_list = []

        # Loop over local nodes to get basis functions
        for i in range(n_local_dofs):
            dfield = element.gbasis(basis.mapping, X_ref, i)[0]
            val_list.append(dfield.value) # (elements,quadratures), value of ith basis function, at quadrature point, in this element
            grad_list.append(dfield.grad) # (dimensions,elements,quadratures), value of the derivative in a direction, of the ith basis function, at quadrature point, in this element

        phi_val = jnp.array(np.stack(val_list)).transpose(1, 2, 0) # eth index is interpolation matrix for element e
        phi_grad = jnp.array(np.stack(grad_list)).transpose(2, 1, 3, 0) #eth index, array at dth index, is interpolation matrix for element e 

        dof_map = basis.element_dofs.T

        return weights,phi_val,phi_grad,dof_map,boundary_dofs

        weights,phi_val,phi_grad,dof_map,boundary_dofs = preprocess(basis)
        all_dofs = np.arange(basis.N)
        interior_dofs = np.setdiff1d(all_dofs, boundary_dofs)
        node_coords_global = jnp.array(mesh.doflocs.T)
        boundary_condition = 0