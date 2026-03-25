'''
This code solves the Poisson equation Nabla u = f expressed in terms of the function u
The Hessian of u is solved in a post-processing (pp) variational problem, because one cannot take directly the second derivative of u (u.dx(i).dx(j)) [this would lead to divergences]


Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]

Examples:

    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/vertex/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_vertex $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/vertices/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk_vertices $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/vertices/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk_vertices_tangent $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/ring_slice/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_slice $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/half_circle_with_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py half_circle_with_line $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/symmetric/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_symmetric $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/ring_with_circle/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_with_circle $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/ring_with_lines/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/symmetric/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/symmetric/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_mirror $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/two_squares_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py two_squares_no_circle $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/lines/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/circles/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_top_bottom/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_symmetric_top_bottom $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_left_right_top_bottom/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_symmetric_left_right_top_bottom $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_ellipse $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_polygon $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/half_circle/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_half_circle $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/3d/ball/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ball $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/3d/box/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py box $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/3d/box_ball/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py box_ball $MESH_PATH $SOLUTION_PATH

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
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }

'''
# test FacetTangent - start
import differential_geometry.boundary.geometry as bgeo
import input_output as io
import solution_paths as solpath

t = bgeo.calc_tangent_cg2(rmsh.lmsh.mesh)
io.full_print(t, 't', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
# test FacetTangent - end
'''

var_pr.solve_vp(vp.F, fsp.u, vp.bcs, fsp.J_u, parameters=params)

var_pr.solve_vp(vp.F_pp, fsp.hess_u, [], fsp.J_hess_u)

prout_bc = importlib.import_module(swi.prout_bc)
