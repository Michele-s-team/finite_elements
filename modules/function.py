import colorama as col
# this import is needed, do not remove it 
import dolfin
from fenics import *
import importlib
import numpy as np
import math
import ufl

i, j, k, l = ufl.indices(4)

msh = importlib.import_module('mesh.utils')

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

    # FIX: Find coordinate columns that start with ':'
    coord_cols = [i for i, col in enumerate(df.columns) if str(col).startswith(':')][:gdim]
    if len(coord_cols) < gdim:
        # Fallback to old behavior if no ':' columns found
        coords_csv = df.iloc[:, value_size:value_size + gdim].to_numpy(dtype=float)
    else:
        coords_csv = df.iloc[:, coord_cols].to_numpy(dtype=float)

    # Get DOF coordinates (one per DOF)
    dof_coords = f.function_space().tabulate_dof_coordinates()
    # Reshape to (n_dofs, gdim)
    dof_coords = dof_coords.reshape((-1, gdim))

    # FIX: For higher-order elements, we need to handle all DOF coordinates directly
    # Build KD-tree on CSV node coords
    tree = cKDTree(coords_csv)

    # Find nearest CSV node for each DOF coordinate
    dist, idx = tree.query(dof_coords, k=1)
    if np.max(dist) > tol:
        print(f"Warning: max coordinate mismatch = {np.max(dist):.3e} (tol={tol:.1e})")

    # Prepare vector of values matching DOF ordering
    reordered = np.zeros(dof_coords.shape[0])

    if value_size == 1:
        # Scalar field case
        for dof_i in range(len(dof_coords)):
            csv_i = idx[dof_i]
            reordered[dof_i] = values_csv[csv_i, 0]
    else:
        # Vector field case - assign components based on DOF ordering
        # For interleaved DOFs: [x0, y0, x1, y1, x2, y2, ...]
        for dof_i in range(len(dof_coords)):
            csv_i = idx[dof_i]
            comp = dof_i % value_size
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
read a field stored in a csv file
Input values: 
- 'file_path': the path to the csv file, including folder, namefile and extension
- 'u': the field where the read values will be stored
- 'type': the type of field to be read, e.g., 'scalar' or 'vector'. In this method, the number of components of the vector needs not match the dimension of the mesh
'''
def read_from_file(file_path, u):

    u_dummy = Function(u.function_space())

    # obtain the number of components of u
    n_components = u.function_space().ufl_element().value_size()

    print(f'number of components = {n_components}')

    class Expression(UserExpression):
        def eval(self, values, x):

            if n_components == 1:
                values[0] = u_dummy(x)
            else:
                for i in range(n_components):
                    values[i] = (u_dummy(x))[i]

        def value_shape(self):
            return (n_components,)

    set_from_file(u_dummy, file_path)
    u.interpolate(Expression(element=u.function_space().ufl_element()))


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

'''
given a field defined on a mesh and a deformation field of the mesh, return the field defined and interpolated on the deformed mesh
Input values: 
    - 'f': the field (scalar, vector or tensor)
    - 'u': the deformation field, defined on the mesh of f

'''
def deform_function(f, u):

    Q = deform_function_space(f.function_space(), u)

    g = Function(Q)
    copy_function_values(f, g)

    return g




'''

given a rectangular mesh and a sub mesh given by its top edge, transfer the values of a field (scalar, vector or tensor) defined on the sub mesh to a function defined on the mesh, setting to zero the values of the mesh function at points not on the edge.
Input values:
    - 'u_sub_mesh': the field defined on the sub mesh (it needs to have the same shape as 'u_mesh')
    - 'u_mesh': the field defined on the mesh
'''

def transfer_sub_mesh_to_mesh(u_sub_mesh, u_mesh):


    Q_mesh = u_mesh.function_space()
    
    # compute the height of the mesh rectangle 
    h = (msh.compute_size(Q_mesh.mesh()))[1]
    

    # Get DOF coordinates for the mesh function space
    mesh_coordinates = Q_mesh.tabulate_dof_coordinates()
    
    # Determine the value shape (scalar, vector, or tensor)
    value_shape = Q_mesh.ufl_element().value_shape()
    value_rank = len(value_shape)
    
    # Calculate total number of components
    if value_rank == 0:
        # Scalar field
        num_components = 1
    elif value_rank == 1:
        # Vector field
        num_components = value_shape[0]
    else:
        # Tensor field (e.g., 2x2 matrix has 4 components)
        num_components = int(np.prod(value_shape))
    
    # For vector spaces, coordinates are repeated for each component
    # We need to evaluate only at unique coordinates
    num_unique_points = len(mesh_coordinates) // num_components
    
    # Create list to store all DOF values (using list for efficiency with extend)
    all_values = []
    
    # Process each unique point
    for i in range(num_unique_points):
        # run through mesh_coordinates with step num_components
        mesh_coord = mesh_coordinates[i * num_components]
        
        # Check if this point is on the edge y = h
        if math.isclose(mesh_coord[1], h):
            # Evaluate the sub_mesh function at x-coordinate
            value = u_sub_mesh(mesh_coord[0])
            
            if num_components == 1:
                # Scalar field - direct assignment
                all_values.append(value)
            else:
                # Vector or tensor field
                # Extend with all components at once (interleaved ordering)
                all_values.extend(np.array(value, dtype=float).flatten())
        else:
            # Point not on edge - add zeros
            if num_components == 1:
                all_values.append(0.0)
            else:
                all_values.extend([0.0] * num_components)
    
    # Set the values in the function
    u_mesh.vector()[:] = np.array(all_values)
        


'''
transfer on a sub mesh a function defined on a mesh, where the mesh is given by a rectangle, and the sub mesh by its top edge. 
Input values: 
    - 'f_mesh': the function defined on the mesh (a scalar, vector, tensor of any shape)
    - 'f_sub_mesh': the function defined on the sub mesh (it needs to have the same shape as 'f_mesh')
    - 'h': the height of the rectangle mesh 
'''
def transfer_mesh_to_sub_mesh(f_mesh, f_sub_mesh, h):
    # Get DOF coordinates
    sub_mesh_dim = f_sub_mesh.function_space().mesh().geometry().dim()
    dof_coords_sub_mesh = f_sub_mesh.function_space().tabulate_dof_coordinates().reshape((-1, sub_mesh_dim))
    
    # Get value shape
    element = f_sub_mesh.function_space().ufl_element()
    value_shape = element.value_shape()
    
    if len(value_shape) == 0:
        value_size = 1
    elif len(value_shape) == 1:
        value_size = value_shape[0]
    else:
        value_size = np.prod(value_shape)
    
    # For tensor/vector spaces, coordinates are repeated for each component
    # We need to evaluate only at unique coordinates
    num_unique_points = len(dof_coords_sub_mesh) // value_size
    
    # Create flat array to store all DOF values
    all_values = []
    
    # Evaluate at each unique coordinate
    for i in range(num_unique_points):
        coord = dof_coords_sub_mesh[i * value_size]  # Take first occurrence of each unique point
        
        val = f_mesh([coord[0], h])
        
        if value_size == 1:
            all_values.append(val)
        else:
            # val is already the full tensor (4 components for 2x2)
            all_values.extend(np.array(val).flatten())
                

    
    # Assign to the submesh function
    f_sub_mesh.vector()[:] = np.array(all_values)
    
    


'''
Compute the average between left and right side ('+' and '-') of a field on an internal mesh domain
Input values: 
- 'f': the field (so far, this method works if 'f' is a scalar or a vector of any dimension, but it does not work if 'f' is a tensor)
Return values: 
- (f('+') + f('-'))/2.0 for a scalar,  as_tensor((((f('+'))[i] + (f('-'))[i])/2.0), (i)) for a vector
'''
def average_dS(f):
    
    shape = f.ufl_shape
    rank = len(shape)
    
    if rank == 0:
        
        return ((f('+') + f('-'))/2.0)
        
    elif rank == 1:
        
        return as_tensor((((f('+'))[i] + (f('-'))[i])/2.0), (i))

    else:
        print(f"{col.Fore.RED}{'Error: called compute average_dS with a tensor, I cannot compute average_dS !'}{col.Style.RESET_ALL}")
     
'''
return the error norm of the difference between two functions. The two functions will be interpolated on a function space with higher degree than the respective function spaces of the two functions, and then the norm of the difference between these two interpolated functions will  be taken  
Input values: 
    - Mandatory: 
        * 'f' and 'g': the two functions
        * 'measure': the measure where the error norm will be computed
    - Optional: 
        * 'delta_function_space_degree': the increment of the degree of the polynomial space. The max of the degree of the space of f and g, will be incremented by 'delta_function_space_degree', and this will give the degree of 'Q_high', the polynomial space where f and g will be interpolated

'''
def error_norm(f, g, measure, delta_function_space_degree=3):
    
    mesh = f.function_space().mesh()    
        
    degree_f = f.function_space().ufl_element().degree()
    degree_g = g.function_space().ufl_element().degree()
        
    Q_high = FunctionSpace(mesh, 'P', max(degree_f, degree_g) + delta_function_space_degree)
    error = Function(Q_high)  
    
    f_high = interpolate(f, Q_high) 
    g_high = interpolate(g, Q_high) 
    
    # Subtract degrees of freedom for the error field 
    error.vector()[:] = g_high.vector().get_local() -  f_high.vector().get_local() 
    error = (error**2)*measure
    
    return sqrt(assemble(error))