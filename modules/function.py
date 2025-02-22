from fenics import *

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

'''

def my_expression_1(x):
    return np.cos(x[0]) * x[1]


class my_expression_2( UserExpression ):
    def eval(self, values, x):

        values[0] = np.cos(x[0]) * x[1]

    def value_shape(self):
        return (1,)

f_test_1 = Function( fsp.Q_z )
fu.set_nodal_values_expression( f_test_1, my_expression_1 )

f_test_2 = interpolate( my_expression_2( element=fsp.Q_z.ufl_element() ), fsp.Q_z )


xdmffile_f_test = XDMFFile( (rarg.args.output_directory) + "/f_test.xdmf" )
xdmffile_f_test.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

xdmffile_f_test.write( f_test_1, 0 )
xdmffile_f_test.write( f_test_2, 0 )
xdmffile_f_test.close()
'''

def set_nodal_values_list(f, list):

    mesh = f.function_space().mesh()

    Q_dummy = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q_dummy.tabulate_dof_coordinates()

    for i in range(Q_dummy.dim()):
        f.vector()[i] = list[i][0]