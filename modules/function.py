import dolfin
from fenics import *
import numpy as np
import math
from ufl import FunctionSpace

import mesh.utils as msh

'''
set the nodal values of f equal to the values taken by the analytical expression 'expression' on the  points of the mesh of f, where expression should be like this

def expression(x):
    return np.cos(x[0]) * x[1]
'''


def set_nodal_values_expression(f, expression):
    mesh = f.function_space().mesh()

    Q_dummy = FunctionSpace(mesh, 'CG', 1)
    coordinates = Q_dummy.tabulate_dof_coordinates()

    for i in range(Q_dummy.dim()):
        f.vector()[i] = expression(coordinates[i])


# set the nodal values of function 'f' according to the list 'list'. This works only if the function space of f is order-1 polynomials
def set_from_list(f, list):
    mesh = f.function_space().mesh()

    Q_dummy = FunctionSpace(mesh, 'CG', 1)
    coordinates = Q_dummy.tabulate_dof_coordinates()

    for i in range(Q_dummy.dim()):
        f.vector()[i] = list[i][0]


def set_from_file(f, filename, constraint=None, tol=1e-12):
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    mesh = f.function_space().mesh()
    gdim = mesh.geometry().dim()
    element = f.function_space().ufl_element()
    value_size = element.value_size()  # number of components per node

    # Read CSV file
    df = pd.read_csv(filename, comment="#")
    ncols = df.shape[1]
    if ncols < value_size + gdim:
        raise ValueError(f"CSV has {ncols} columns but expected at least {value_size + gdim}")

    # Extract values and coords from CSV
    values_csv = df.iloc[:, :value_size].to_numpy(dtype=float)  # (n_nodes_csv, value_size)
    coords_csv = df.iloc[:, value_size:value_size + gdim].to_numpy(dtype=float)  # (n_nodes_csv, gdim)

    # Get DOF coordinates (one per DOF)
    dof_coords = f.function_space().tabulate_dof_coordinates()
    # Reshape to (n_dofs, gdim)
    dof_coords = dof_coords.reshape((-1, gdim))

    # Extract unique node coords by taking every value_size-th DOF coordinate
    n_nodes = dof_coords.shape[0] // value_size
    node_coords = dof_coords[0: n_nodes * value_size: value_size]

    # Build KD-tree on CSV node coords
    tree = cKDTree(coords_csv)

    # Find nearest CSV node for each mesh node coordinate
    dist, idx = tree.query(node_coords, k=1)
    if np.max(dist) > tol:
        print(f"Warning: max coordinate mismatch = {np.max(dist):.3e} (tol={tol:.1e})")

    # Prepare vector of values matching DOF ordering (component interleaved)
    reordered = np.zeros(dof_coords.shape[0])

    for i_node in range(n_nodes):
        csv_i = idx[i_node]
        for comp in range(value_size):
            dof_i = i_node * value_size + comp
            reordered[dof_i] = values_csv[csv_i, comp]

    if reordered.size != f.vector().size():
        raise ValueError(
            f"Mismatch: CSV provides {reordered.size} DOF-values, "
            f"but function requires {f.vector().size()}"
        )

    # Assign to function vector
    f.vector()[:] = reordered

    if constraint is not None:
        constraint.apply(f.vector())


'''
given a function space and its mesh, return a function space on the deformed mesh, deformed according to a displacement field
Input values:
- 'Q': the function space
- 'u': the displacement field
Return values:
- the new function space on the deformed mesh
'''


def deform_function_space(Q, u):
    deformed_mesh = msh.deform_mesh(Mesh(Q.mesh()), u)

    # Extract the features of the vector space Q
    element = Q.ufl_element()
    family = element.family()
    cell = element.cell()
    shape = element.value_shape()
    degree = Q.ufl_element().degree()

    # Construct the new element with the same shape
    if shape == ():  # scalar
        element = FiniteElement(family, cell, degree)
    elif len(shape) == 1:  # vector
        element = VectorElement(family, cell, degree, dim=shape[0])
    elif len(shape) == 2:  # tensor
        element = TensorElement(family, cell, degree, shape=shape)
    else:
        raise ValueError(f"Unsupported value shape: {shape}")

    return dolfin.FunctionSpace(deformed_mesh, element)


'''
copy the values of a function (nodal values, values within the triangles, etc.) to another function. This works for scalars, vectors, tensors. 
Input values:
- 'f_in', 'f_out': source and destination function
'''


def copy_function_values(f_in, f_out):
    f_out.vector()[:] = f_in.vector()[:]


def deform_function(f, u):
    Q = deform_function_space(f.function_space(), u)
    # print(f'type of Q = {type(Q)}')  # should be <class 'dolfin.cpp.function.FunctionSpace'>

    g = Function(Q)
    copy_function_values(f, g)

    return g


def transfer_sub_mesh_to_mesh(u_sub_mesh, Q_mesh, Q_sub_mesh, h):
    u_sub_mesh_on_mesh = Function(Q_mesh)

    # Get DOF coordinates for both function spaces
    mesh_coordinagtes = Q_mesh.tabulate_dof_coordinates()
    sub_mesh_coordinates = Q_sub_mesh.tabulate_dof_coordinates()  # DOFs, not just vertices!

    # initialize all the values to 0
    dof_values = np.zeros(Q_mesh.dim())

    for mesh_id, mesh_coord in enumerate(mesh_coordinagtes):


        if math.isclose(mesh_coord[1], h):
            # print(f'point on edge is TRUE for mesh_coord = {mesh_coord}')
            dof_values[mesh_id] = u_sub_mesh(mesh_coord[0])


    u_sub_mesh_on_mesh.vector()[:] = dof_values
    return u_sub_mesh_on_mesh
