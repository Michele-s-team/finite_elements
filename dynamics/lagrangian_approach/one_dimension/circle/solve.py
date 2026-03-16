'''
This code solves for the dynamics of a circular, one-dimensional shape embedded in two dimensions, under the influence of an advecting velocity field in the two-dimensional space. The one-dimensional shape is parameterized by means of a coordinate living on a one-dimensional, line mesh. 

Run with 
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/lagrangian_approach/one_dimension/circle/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 solve.py circle $MESH_PATH $SOLUTION_PATH;

'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as sys_io
import mesh.load as lmsh
import parameters.read.solution as rpam
import solution_paths as solpath
import switch_problem as swi



fsp = importlib.import_module(swi.fsp)
prout_bc = importlib.import_module(swi.prout_bc)
prout_sol = importlib.import_module(swi.prout_sol)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

#set initial profiles
fsp.u_n.interpolate( vp.u_n_1_expression( element=fsp.Q.ufl_element() ))
fsp.u_n_1.interpolate( vp.u_n_1_expression( element=fsp.Q.ufl_element() ))

# set solver parameters
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

# print the reference configuration 
sys_io.full_print(fsp.ys, 'ys', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'vector')

# Time-stepping
t = 0
for step in range(rpam.parameters['N']):

    print("\n* step = ", step, "\n",flush=True)

    # Update current time
    t += vp.dt

    vp = importlib.import_module(swi.vp)

    J = derivative(vp.F, fsp.u_n, fsp.J_u_n)
    problem = NonlinearVariationalProblem(vp.F, fsp.u_n, vp.bcs, J)
    solver = NonlinearVariationalSolver(problem)

    # J_pp = derivative(vp.F_pp, fsp.grad_u, fsp.J_grad_u)
    # problem_pp = NonlinearVariationalProblem(vp.F_pp, fsp.grad_u, [], J_pp)
    # solver_pp = NonlinearVariationalSolver(problem_pp)


    solver.parameters.update(params)
    # solver_pp.parameters.update(params)

    # solve original problem
    solver.solve()      

    # solve post-processing problem
    # solver_pp.solve()
    
    prout_bc.print_bcs()

    if(step % rpam.parameters['print_out_stride'] == 0):
        prout_sol.print_solution(step)


    #update the solution
    fsp.u_n_1.assign(fsp.u_n)



prout_bc = importlib.import_module(swi.prout_bc)
prout_sol = importlib.import_module(swi.prout_sol)

