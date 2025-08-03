'''
This code solves the Poisson equation  Nabla u = f expressed in terms of the function u and v_i = \partial_i u

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u_v/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk $MESH_PATH $SOLUTION_PATH

'''

import colorama as col
from fenics import *
import importlib

import sys
import ufl

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import input_output as io
import load_mesh as lmsh
import mesh as msh
import solution_paths as solpath
import switch_problem as swi

i, j, k, l = ufl.indices(4)

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

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
solver.solve()

# print out the solution
u_output, v_output = fsp.psi.split(deepcopy=True)

io.full_print(u_output, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')
io.full_print(v_output, 'v', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'vector')

# check if the boundary conditions are satisfied
print("BCs check: ")
print(f"\t\t<<(u - u_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(n.v-n.v_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[i] * v_output[i], bgeo.facet_normal[i] * fsp.v_exact[i], rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

# check if the FE solution agrees with the exact one
print("Comparison with exact solution: ")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<|v - v_exact|^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(geo.ufl_norm(v_output - fsp.v_exact), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
