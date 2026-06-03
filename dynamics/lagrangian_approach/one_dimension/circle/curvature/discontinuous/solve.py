"""
This code solves for the mean curvature of a current shape defined with the lagrangian approach as the deformation of a reference shape. The mean curvature mu obtained is a field defined on the full mesh, and if y_s \in \partial \Omega^y_circle, we have

 mu(y_s) = H_s

 where H_s is the mean curvature of the current shape computed at point x_s, with respect to a unit normal pointing into the shape. 
 The two problems are :
    - 'square_shape_line_a': here the function y(s) and dy/ds is set from an analytical expression
    - 'square_shape_line_b': here the function y(s) and dy/ds is obtained by fitting shape coordinates

NOTE: This approach works only if the parametric curve y(s) vs. s has no overhangs (this is because the inversion s_y(x) is used)

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:

    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/lagrangian_approach/one_dimension/circle/curvature/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_a $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/lagrangian_approach/one_dimension/circle/curvature/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_b $MESH_PATH $SOLUTION_PATH

"""

import dolfin
from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import switch_problem as swi
import variational_problem.utils as var_pr

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


dolfin.parameters["form_compiler"]["quadrature_degree"] = 10


var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi)


vp = importlib.import_module(swi.prout_sol)

   