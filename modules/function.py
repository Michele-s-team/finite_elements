import colorama as col
# this import is needed, do not remove it 
import dolfin
from fenics import *
import importlib
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
import ufl

import calculus as cal
import constants.utils as const
import input_output as io


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


'''
set a field from the data contained in a csv file
Input values: 
    * Mandatory: 
        - `f`: the field (scalar, vector, tensor)
        - `filename`: the path, filename and extension of the csv file
    * Optional: 
        - `tol`: the tolerance used to find the nearest CSV node for each DOF coordinate
'''
def set_from_file(f, filename, tol=1e-12):


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
given a rectangular mesh and a sub mesh given by its top edge (which can be any one-dimensional manifold, not necessarily a line), transfer the values of a field (scalar, vector or tensor) defined on the sub mesh to a function defined on the mesh, setting to zero the values of the mesh function at points not on the edge.
Input values:
    * Mandatory: 
        - 'u_sub_mesh': the field defined on the sub mesh (it needs to have the same shape as 'u_mesh')
        - 'u_mesh': the field defined on the mesh
        - 'mesh_path': the path where the mesh is stored

    * Optional:
        - 'tol' (const.epsilon): the tolerance used to assess distances
'''

def transfer_sub_mesh_to_mesh(u_sub_mesh, u_mesh, mesh_path,
                              tol=const.epsilon):



    Q_mesh = u_mesh.function_space()
    
    '''
    read all vertices which belong to edges tagged with ID 'sub_mesh_1_id' and store them into `sub_mesh_1_vertices`
    `sub_mesh_vertices` is an ordered list of the coordinates of the vertices in the mesh which belong to the sub mesh 
    '''
    mesh_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, 'mesh_metadata.csv')) 
    sub_mesh_vertices = mesh_parameters['curve_coordinates']

    '''
    compute the arc length along  the sub mesh: arc_length_tab[i] = [cumulative arc length along the sub mesh curve obtained from its beginning until sub_mesh_vertices included]
    '''
    arc_length = 0
    arc_length_tab = [0]
    for i in range(1, len(sub_mesh_vertices)):

        arc_length += np.linalg.norm(np.subtract(sub_mesh_vertices[i], sub_mesh_vertices[i-1]))
        arc_length_tab.append(arc_length)

    

    
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

    # Get DOF coordinates for the mesh function space
    dof_coordinates = Q_mesh.tabulate_dof_coordinates()
    n_dofs = Q_mesh.dim()
    n_nodes = n_dofs // num_components

    
    # Create list to store all DOF values (using list for efficiency with extend)
    u_mesh_values = np.zeros(n_dofs)

    
    # Process each unique point
    for node in range(n_nodes):
        # run through mesh_coordinates with step num_components

        coordinate = dof_coordinates[node * num_components]

        for i in range(1, len(sub_mesh_vertices)):
            # run through `sub_mesh_vertices` to find whether `node` belongs to the sub mesh

            if cal.point_on_segment(np.array(coordinate), np.array(sub_mesh_vertices[i-1]), np.array(sub_mesh_vertices[i]), tol):
                #  `node` lies on the segment in between two verices in `sub_mesh_vertices` -> it belongs to the sub mesh 

                # arc length at the DOF = cumulative length up to v_{i-1} + distance along this segment
                s = arc_length_tab[i-1] + np.linalg.norm(
                        np.subtract(coordinate, sub_mesh_vertices[i-1]))

                # compute u_sub_mesh at the arc length `s`
                u_sub_mesh_value = np.array(u_sub_mesh(s), dtype=float).flatten()

                # assign the compute value of `u_sub_mesh` to u_mesh_values
                for j in range(num_components):

                    u_mesh_values[num_components*node + j] = u_sub_mesh_value[j]

                break

               
    # set the values in u_mesh
    u_mesh.vector().set_local(u_mesh_values)
    u_mesh.vector().apply("insert")
        


'''
transfer on a sub mesh a function defined on a mesh, where the mesh is given by a rectangle, and the sub mesh by its top edge and it needs not be a straight line. 
Input values: 
    * Mandatory:
        - 'u_mesh': the function defined on the mesh (a scalar, vector, tensor of any shape)
        - 'u_sub_mesh': the function defined on the sub mesh (it needs to have the same shape as 'f_mesh')
    * Optional:
        - 'tol' (const.epsilon): the tolerance used to assess distances
'''
def transfer_mesh_to_sub_mesh(u_mesh, u_sub_mesh, mesh_path, tol = const.epsilon):

    # this is needed in case `u_mesh` is evaluated at point slightly outside its mesh
    u_mesh.set_allow_extrapolation(True)


    Q_sub_mesh = u_sub_mesh.function_space()

    '''
    read all vertices which belong to edges tagged with ID 'sub_mesh_1_id' and store them into `sub_mesh_1_vertices`
    `sub_mesh_vertices` is an ordered list of the coordinates of the vertices in the mesh which belong to the sub mesh 
    '''
    mesh_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, 'mesh_metadata.csv')) 
    sub_mesh_vertices = mesh_parameters['curve_coordinates']

    '''
    compute the arc length along  the sub mesh: arc_length_tab[i] = [cumulative arc length along the sub mesh curve obtained from its beginning until sub_mesh_vertices included]
    '''
    arc_length = 0
    arc_length_tab = [0]
    for i in range(1, len(sub_mesh_vertices)):

        arc_length += np.linalg.norm(np.subtract(sub_mesh_vertices[i], sub_mesh_vertices[i-1]))
        arc_length_tab.append(arc_length)


    # Determine the value shape (scalar, vector, or tensor)
    value_shape = Q_sub_mesh.ufl_element().value_shape()
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

    # Get DOF coordinates
    dof_coordinates = Q_sub_mesh.tabulate_dof_coordinates()
    n_dofs = Q_sub_mesh.dim()
    n_nodes = n_dofs // num_components

    # Create list to store all DOF values (using list for efficiency with extend)
    u_sub_mesh_values = np.zeros(n_dofs)

    
    # Evaluate at each unique coordinate
    for node in range(n_nodes):

        coordinate = dof_coordinates[node * num_components]  # Take first occurrence of each unique point

        '''
        convert `coord[0]` into an arclength along the mesh: find the pair of entries in `arc_length_tab` that bracked coord[0]
        '''

        # print(f'* coordinate[0] = {coordinate[0]}')

        for j in range(len(arc_length_tab)-1):

            if (coordinate[0] > arc_length_tab[j] - tol) and  (coordinate[0] < arc_length_tab[j+1] + tol):
                # `coordinate[0]` falls within arc_length_tab[j] and arc_length_tab[j+1] -> break the loop and store j
                break

        '''
        the loop above returns j such that arc_length_tab[j] < coord[0] < arc_length_tab[j+1]
        '''
        # print(f'* j = {j}')

        mesh_coordinate = np.add(sub_mesh_vertices[j], np.multiply((coordinate[0] - arc_length_tab[j])/(arc_length_tab[j+1] - arc_length_tab[j]), np.subtract(sub_mesh_vertices[j+1], sub_mesh_vertices[j])))

        u_mesh_value = np.array(u_mesh(mesh_coordinate), dtype=float).flatten()
        
        # assign the compute value of `u_sub_mesh` to u_mesh_values
        for j in range(num_components):

            u_sub_mesh_values[num_components*node + j] = u_mesh_value[j]
                
    
    # set the values in u_mesh
    u_sub_mesh.vector().set_local(u_sub_mesh_values)
    u_sub_mesh.vector().apply("insert")
    
    
'''
given a sub mesh a and a sub mesh b obtained from a by means of a displacement field, transfer a field (scalar, vector, tensor) on sub mesh a onto sub mesh b
Input values: 
    * Mandatory:
        - `u_sub_mesh_a`: the field on sub mesh a
        - `u_sub_mesh_b`: the field on sub mesh b
        - `u`: the deformation field that relates  mesh a to sub mesh b
    * Optional:
        - 'tol' (const.epsilon): the tolerance used to assess distances
'''
def transfer_sub_mesh_to_sub_mesh(u_a, u_b, u, mesh_a_path, tol = const.epsilon):

    Q_b = u_b.function_space()


    '''
    read all vertices in mesh a which belong to edges tagged with ID 'sub_mesh_1_id' and store them into `vertices_a`
    `vertices_a` is an ordered list of the coordinates of the vertices in  mesh a which belong to the sub mesh 
    '''
    mesh_a_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_a_path, 'mesh_metadata.csv')) 
    mesh_a_vertices = mesh_a_parameters['curve_coordinates']

    '''
    compute the arc length along sub mesh a: arc_length_a_tab[i] = [cumulative arc length along the sub mesh a curve obtained from its beginning until vertices_a[i] included]
    '''
    arc_length_a = 0
    arc_length_a_tab = [0]
    for i in range(1, len(mesh_a_vertices)):

        arc_length_a += np.linalg.norm(np.subtract(mesh_a_vertices[i], mesh_a_vertices[i-1]))
        arc_length_a_tab.append(arc_length_a)


    '''
    compute the arc length along sub mesh a, deformed onto sub mesh b: arc_length_b_tab[i] = [cumulative arc length along the sub mesh a curve deformed into b, obtained from its beginning until sub_mesh_mesh_a_vertices[i] included]
    '''
    arc_length_a_to_b = 0
    arc_length_a_to_b_tab = [0]
    for i in range(1, len(mesh_a_vertices)):

        arc_length_a_to_b += np.linalg.norm(np.subtract(
            np.add(mesh_a_vertices[i], u(mesh_a_vertices[i])), 
            np.add(mesh_a_vertices[i-1], u(mesh_a_vertices[i-1]))
            ))
        arc_length_a_to_b_tab.append(arc_length_a_to_b)


    # Determine the value shape (scalar, vector, or tensor)
    value_shape = Q_b.ufl_element().value_shape()
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

    # Get DOF coordinates
    dof_coordinates_b = Q_b.tabulate_dof_coordinates()
    n_dofs_b = Q_b.dim()
    n_nodes_b = n_dofs_b // num_components


    # Create list to store all DOF values (using list for efficiency with extend)
    u_b_values = np.zeros(n_dofs_b)

    # Evaluate at each unique coordinate
    for node in range(n_nodes_b):

        # Take first occurrence of each unique point along b
        coordinate_b = dof_coordinates_b[node * num_components][0]

        '''
        convert `coordinate_b` into an arclength along a by taking into account the deformation field `u` that brings a into b: find the pair of entries in `arc_length_a_to_b_tab` that bracked coordinate_b
        '''

        # print(f'* coordinate[0] = {coordinate[0]}')

        for j in range(len(arc_length_a_to_b_tab)-1):

            if (coordinate_b > arc_length_a_to_b_tab[j] - tol) and  (coordinate_b < arc_length_a_to_b_tab[j+1] + tol):
                # `coordinate[0]` falls within arc_length_b_tab[j] and arc_length_b_tab[j+1] -> break the loop and store j
                break

        '''
        the loop above returns j such that arc_length_b_tab[j] < coord[0] < arc_length_b_tab[j+1]
        '''
        # print(f'* j = {j}')

        coordinate_a = arc_length_a_tab[j] + (coordinate_b - arc_length_a_to_b_tab[j]) / (arc_length_a_to_b_tab[j+1] - arc_length_a_to_b_tab[j]) * (arc_length_a_tab[j+1] - arc_length_a_tab[j])

        u_a_value = np.array(u_a(coordinate_a), dtype=float).flatten()
        
        # assign the compute value of `u_sub_mesh` to u_mesh_values
        for j in range(num_components):

            u_b_values[num_components*node + j] = u_a_value[j]
                
    
    # set the values in u_mesh
    u_b.vector().set_local(u_b_values)
    u_b.vector().apply("insert")
    



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



'''
class defining the identity function expression in two dimensions
Input values:
    - 'x': [x_0, x_1] the input coordinates
Return values: 
    - 'x'
'''

class identity_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0] 
        values[1] = x[1] 
        
    def value_shape(self):
        return (2,)