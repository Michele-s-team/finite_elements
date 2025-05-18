from fenics import *

import input_output as io

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
