'''
This code solves for the curvature of a periodic shape, defined on a 2d shape and laid down flat on a 1d mesh 

The problem has three meshes:
- mesh[0]: a 2d mesh given by the box, including the shape in it. This is divided into 
    * sub_mesh[0]: the shape
    * sub_mesh[1]: the surface between the shape boundary and the box. 
- mesh[1]: a 1d mesh given by a line (the boundary of the shape obstacle laid flat on a line)

Here, only mesh[1] is used. 

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
    
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/lagrangian_approach/one_dimension/circle/curvature/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_a $MESH_PATH $SOLUTION_PATH
 '''

import dolfin
from fenics import *
import importlib
import numpy as np
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import input_output as io
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi
import variational_problem.utils as var_pr



dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 


# create the solution metadata and write it into the output directory 
metadata = rpam.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, "solution_metadata.csv"), metadata)

# # parameters with SNES method
# 
params = {
    'nonlinear_solver': 'newton',
    'snes_solver': {
        'linear_solver': 'superlu',
        'line_search': 'bt',  # backtracking line search
        'absolute_tolerance': 1e-6,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 1000000,
        'report': True,
    }
}


'''
PETScOptions.clear()
PETScOptions.set('snes_type', 'newtontr')
PETScOptions.set('snes_atol', 1e-6)     # Stricter absolute tolerance
PETScOptions.set('snes_rtol', 1e-6)     # Stricter relative tolerance
PETScOptions.set('snes_stol', 1e-8)      # Keep step tolerance same
PETScOptions.set('snes_max_it', 100000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 1000000)         # Increase function evaluation limit
'''


import function_spaces as fsp
rmsh = importlib.import_module(swi.rmsh)

vp_I = importlib.import_module(swi.vp_I)

class U_expression(UserExpression):
    def eval(self, values, x):
        
        values[0] = mesh_parameters['d'] * np.cos(2*np.pi * 3 * x[0]/rmsh.lmsh.mesh_parameters[1]["L"]) * np.cos(2*np.pi * x[0]/rmsh.lmsh.mesh_parameters[1]["L"])
        values[1] = mesh_parameters['d'] * np.cos(2*np.pi * 3 * x[0]/rmsh.lmsh.mesh_parameters[1]["L"]) * np.sin(2*np.pi * x[0]/rmsh.lmsh.mesh_parameters[1]["L"])

    def value_shape(self):
        return (2,)
    
fsp.U.interpolate(U_expression(element=fsp.Q_U.ufl_element()))

io.full_print(project(fsp.ys + fsp.U, fsp.Q_U), 'X', \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


class nu_dpsi_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0                       
        values[1] = -np.pi/2   
    def value_shape(self):
        return (2,)

fsp.nu_and_dpsi.interpolate(
    nu_dpsi_expression(element=fsp.Q_nu_and_dpsi.ufl_element())
)

var_pr.solve_vp(vp_I.F_nu_psi, fsp.nu_and_dpsi, vp_I.bcs_nu_and_dpsi, fsp.J_nu_and_dpsi, parameters=params)

nu_output, dpsi_output = fsp.nu_and_dpsi.split(deepcopy=True)

io.full_print(nu_output, 'nu', \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
io.full_print(dpsi_output, 'dpsi', \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


var_pr.solve_vp(vp_I.F_mu, fsp.mu, vp_I.bcs_mu, fsp.J_mu, parameters=params)


io.full_print(fsp.mu, 'mu', \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


