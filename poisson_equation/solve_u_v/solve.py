'''
This code solves the Poisson equation  Nabla u = f expressed in terms of the function u and v_i = \partial_i u

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk $MESH_PATH $SOLUTION_PATH

'''

import colorama as col
from fenics import *
import importlib
import runtime_arguments as rarg
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import boundary_geometry as bgeo
import function_spaces as fsp
import input_output as io
import mesh as msh
import switch_problem as swi

i, j, k, l = ufl.indices( 4 )


rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

'''
L = 1
h = 1
alpha = 1e2
function_space_degree = 4
'''

# parser = argparse.ArgumentParser()
# parser.add_argument( "input_directory" )
# parser.add_argument( "output_directory" )
# args = parser.parse_args()

xdmffile_u = XDMFFile((rarg.args.output_directory) + "/u.xdmf")
xdmffile_u.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_v = XDMFFile((rarg.args.output_directory) + "/v.xdmf")
xdmffile_v.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_check = XDMFFile((rarg.args.output_directory) + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

# read the mesh
'''
mesh = Mesh()
xdmf = XDMFFile( mesh.mpi_comm(), (rarg.args.input_directory) + "/triangle_mesh.xdmf" )
xdmf.read( mesh )

# radius of the smallest cell in the mesh
r_mesh = mesh.hmin()

print( f"Mesh radius = {r_mesh}" )

# read the triangles
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() )
with XDMFFile( (rarg.args.input_directory) + "/triangle_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
cf = cpp.mesh.MeshFunctionSizet( mesh, mvc )
xdmf.close()

# read the lines
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() - 1 )
with XDMFFile( (rarg.args.input_directory) + "/line_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
sf = cpp.mesh.MeshFunctionSizet( mesh, mvc )
xdmf.close()
'''

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
# boundary = 'on_boundary'
# boundary_lr = 'near(x[0], 0) || near(x[0], 2.2)'
# boundary_tb = 'near(x[1], 0) || near(x[1], 0.41)'
# CHANGE PARAMETERS HERE

# read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`

# #  norm of vector x
# def my_norm(x):
#     return (sqrt( np.dot( x, x ) ))


# test for surface elements
'''
dx = Measure( "dx", domain=mesh, subdomain_data=cf, subdomain_id=1 )
ds_l = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3 )
ds_t = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=4 )
ds_b = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=5 )
ds = ds_l + ds_r + ds_t + ds_b
'''

# a function space used solely to define f_test_ds
# Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
# f_test_ds = Function( Q_test )


# analytical expression for a  scalar function used to test the ds
# class FunctionTestIntegrals( UserExpression ):
#     def eval(self, values, x):
#         c_test = [0.3, 0.76]
#         r_test = 0.345
#         values[0] = cos( my_norm( np.subtract( x, c_test ) ) - r_test ) ** 2.0
#
#     def value_shape(self):
#         return (1,)
#

# f_test_ds.interpolate( FunctionTestIntegrals( element=Q_test.ufl_element() ) )
#
# # print out the integrals on the volume and  surface elements and compare them with the exact values to double check that the elements are tagged correctly
# msh.test_mesh_integral(0.501508, f_test_ds, dx, '\int f dx')
#
# msh.test_mesh_integral(0.373168, f_test_ds, ds_l, '\int_l f ds')
# msh.test_mesh_integral(0.00227783, f_test_ds, ds_r, '\int_r f ds')
# msh.test_mesh_integral(1.36562, f_test_ds, ds_t, '\int_t f ds')
# msh.test_mesh_integral(1.02837, f_test_ds, ds_b, '\int_b f ds')
#

# n = FacetNormal( mesh )


# CHANGE PARAMETERS HERE


J = derivative(vp.F, fsp.psi, fsp.J_uv)
problem = NonlinearVariationalProblem(vp.F, fsp.psi, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)
# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update(params)

# solve the variational problem
solver.solve()

# print out the solution
u_output, v_output = fsp.psi.split(deepcopy=True)

xdmffile_u.write(u_output, 0)
xdmffile_v.write(v_output, 0)

io.print_scalar_to_csvfile(u_output, (rarg.args.output_directory) + "/u.csv");
io.print_vector_to_csvfile(v_output, (rarg.args.output_directory) + "/v.csv");

# check if the boundary conditions are satisfied
print("BCs check: ")
print(f"\t\t<<(u - u_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(n.v-n.v_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[i] * fsp.v_output[i], bgeo.facet_normal[i] * fsp.v_exact[i], rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

# check if the FE solution agrees with the exact one
print("Comparison with exact solution: ")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<|v - v_exact|^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(sqrt((fsp.v_output[i] - fsp.v_exact[i]) * (fsp.v_output[i] - fsp.v_exact[i])), Constant(0), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
