'''
This code solves the dynamics of a fluid in a box (A) with a fluid obstacle (B) in the box. 

The problem has three meshes:
- mesh[0]: a 2d mesh given by the box, including the disk in it. This is divided into 
    * sub_mesh[0]: the disk
    * sub_mesh[1]: the surface between the disk boundary and the box. 
- mesh[1]: a 1d mesh given by a line (the boundary of the circular obstacle laid flat on a line)

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
    
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/disk_line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/fluid_obstacle/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_disk_line_a $MESH_PATH $SOLUTION_PATH
 '''

import dolfin
from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

dt = rpam.parameters['T'] / rpam.parameters['N']

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


dolfin.parameters["form_compiler"]["quadrature_degree"] = 10


# load all variational problems
vp_I = importlib.import_module(swi.vp_I)

io.full_print(fsp.ys, 'ys', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path, \
              lmsh.mesh[1], 'vector')

# FILL IN HWERE: set the initial profiles from analytical expressions
# REMEMBER TO TRANSFER FUNCTIONS DURING TIME ITERATION


print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters['N']):
    # Update current time
    t += dt
    step += 1

    # step 1): solve I problem
    print('Solving I problem ...', flush=True)

    # project v_square_n_1 of the fluid in the square onto (mesh[1])
    msh.transfer_circle_to_line(fsp.v_square_n_1, fsp.v_square_n_1_0_1_on_1, lmsh.mesh_parameters[0]['c_r'], lmsh.mesh_parameters[0]['r'], lmsh.mesh_parameters[0]['N'])
    
    vp_I = importlib.reload(vp_I)

    J_I = derivative(vp_I.F_U, fsp.U_n_12, fsp.J_U)
    problem_I = NonlinearVariationalProblem(vp_I.F_U, fsp.U_n_12, vp_I.bcs, J_I)
    solver_I = NonlinearVariationalSolver(problem_I)
    solver_I.parameters.update(params)
    solver_I.solve()

    print('... done.', flush=True)


    # step 2): solve disk fluid problem
    print('Solving disk fluid problem ...', flush=True)

    print('... done.', flush=True)




print("... done.", flush=True)
