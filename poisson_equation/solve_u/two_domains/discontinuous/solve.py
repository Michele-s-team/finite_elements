'''
This code solves the Poisson equation Nabla u = f expressed in terms of the function u, on a mesh (mesh[0]) given by a rectangle with a shape in it. Both the rectangle and the shape are meshed inside. 

I use a discontinuous Galerkin function space, by allowing for a jump of grad u across the shape boundary. 

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_a $MESH_PATH $SOLUTION_PATH

'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import switch_problem as swi
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-10,
                  'relative_tolerance': 1e-10,
                  'maximum_iterations': 1000000,
                #   'relaxation_parameter': 0.95,
              }
          }


'''
import geometry.utils as geo
import numpy as np

x = [0.5 + 0.21 * np.cos(np.pi/4), 0.5 +  0.21 * np.cos(np.pi/4)]

print(f'in polygon = {geo.in_polygon(x, rmsh.lmsh.mesh_parameters[0]["shape_coordinates"])}')
'''

var_pr.solve_vp(vp.F, fsp.u, vp.bcs, fsp.J_u, parameters=params)
prout_bc = importlib.import_module(swi.prout_bc)
