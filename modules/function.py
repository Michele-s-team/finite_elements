from fenics import *
from ufl import FunctionSpace

import input_output as io
import mesh as msh

'''
set the nodal values of f equal to the values taken by the analytical expression 'expression' on the  points of the mesh of f, where expression should be like this

def expression(x):
    return np.cos(x[0]) * x[1]
'''
def set_nodal_values_expression(f, expression):

    mesh = f.function_space().mesh()

    Q_dummy = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q_dummy.tabulate_dof_coordinates()

    for i in range(Q_dummy.dim()):
        f.vector()[i] = expression(coordinates[i])


#set the nodal values of function 'f' according to the list 'list'. This works only if the function space of f is order-1 polynomials
def set_from_list(f, list):

    mesh = f.function_space().mesh()

    Q_dummy = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q_dummy.tabulate_dof_coordinates()

    for i in range(Q_dummy.dim()):
        f.vector()[i] = list[i][0]



#set nodal values of function 'f', defined on a 2d mesh, according to the nodal values written in the csv file 'filename'. . This works only if the function space of f is order-1 polynomials
def set_from_file(f, filename):
    set_from_list( f, io.read_scalar_from_csvfile( filename ) )


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

    return FunctionSpace(deformed_mesh, element)

'''
copy the values of a function (nodal values, values within the triangles, etc.) to another function
Input values:
- 'f_in', 'f_out': source and destination function
'''
def copy(f_in, f_out):
    f_out.vector()[:] = f_in.vector()[:]


