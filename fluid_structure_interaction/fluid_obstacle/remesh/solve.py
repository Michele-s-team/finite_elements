'''
This code solves the dynamics of a fluid in a box (A) with a fluid obstacle (B) in the box. 

The problem has three meshes:
- mesh[0]: a 2d mesh given by the box, including the shape in it. This is divided into 
    * sub_mesh[0]: the shape
    * sub_mesh[1]: the surface between the shape boundary and the box. 
- mesh[1]: a 1d mesh given by a line (the boundary of the shape obstacle laid flat on a line)

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
    
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/fluid_obstacle/remesh/solution"; rm -rf $MESH_PATH; mkdir $MESH_PATH; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_a $MESH_PATH $SOLUTION_PATH
 '''

import dolfin
from fenics import *
import importlib
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import input_output as io
# import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi
import variational_problem.utils as var_pr



dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 

dt = rpam.parameters['T'] / rpam.parameters['N']

# create the solution metadata and write it into the output directory 
metadata = rpam.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, "solution_metadata.csv"), metadata)

# parameters with SNES method
# 
params = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'superlu',
        'line_search': 'bt',  # backtracking line search
        'absolute_tolerance': 1e-6,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 1000000,
        'report': True,
    }
}

PETScOptions.clear()
PETScOptions.set('snes_type', 'newtontr')
PETScOptions.set('snes_atol', 1e-12)     # Stricter absolute tolerance
PETScOptions.set('snes_rtol', 1e-12)     # Stricter relative tolerance
PETScOptions.set('snes_stol', 1e-8)      # Keep step tolerance same
PETScOptions.set('snes_max_it', 100000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 1000000)         # Increase function evaluation limit
# 

class v_sq_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    
class v_di_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    
class sigma_di_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)
    
class sigma_sq_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)

# coordinates of the shape when the shape lies flat (theta_ref = 0)
shape_coordinates_0 = rpam.parameters['shape_coordinates_0']

shape_coordinates = shape_coordinates_0

# generate the mesh with the shape and write theta_ref into its mesh_metadata
msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)



import function_spaces as fsp
import print_out_solution as pr_sol

# 1 set initial profiles
# 1.1 set for square
fsp.v_square_n_1.interpolate(v_sq_0_expression(element=fsp.Q_v_square.ufl_element()))
fsp.v_square_n_2.assign(fsp.v_square_n_1)

fsp.sigma_square_n_12.interpolate(sigma_sq_0_expression(element=fsp.Q_sigma_square.ufl_element()))
fsp.sigma_square_n_32.assign(fsp.sigma_square_n_12)

# 1.2 set for disk
fsp.v_disk_n_1.interpolate(v_di_0_expression(element=fsp.Q_v_disk.ufl_element()))
fsp.v_disk_n_2.assign(fsp.v_disk_n_1)

fsp.sigma_disk_n_12.interpolate(sigma_di_0_expression(element=fsp.Q_sigma_disk.ufl_element()))
fsp.sigma_disk_n_32.assign(fsp.sigma_disk_n_12)


'''
# test map_1d_to_2d - start

import csv
N=200

mesh_1_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, f'mesh_{1}', 'mesh_metadata.csv')) 
print(f'L = {mesh_1_parameters["L"]}')


filename = rarg.args.output_directory + '/test.csv'
os.makedirs(os.path.dirname(filename), exist_ok=True)

csvfile = open(filename, 'a', newline='')
fieldnames = [ \
    "x", \
    "p:0", \
    "p:1"
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

for i in range(N):

    x = mesh_1_parameters['L'] * i/(N-1)
    f = msh.map_1d_to_2d(x, os.path.join(rarg.args.input_directory, f'mesh_{0}'), mesh_parameters['shape_id'])

    writer.writerows([{ \
        fieldnames[0]: \
            x, \
        fieldnames[1]: \
            f[0],\
        fieldnames[2]: \
            f[1]
    }])
    csvfile.flush()

csvfile.close()

# test map_1d_to_2d - end
'''


# fist load of modules
import differential_geometry.manifold.geometry as geo
import differential_geometry.boundary.geometry as bgeo
rmsh = importlib.import_module(swi.rmsh)

vp_I = importlib.import_module(swi.vp_I)
vp_D = importlib.import_module(swi.vp_D)
vp_fl_di = importlib.import_module(swi.vp_fluid_di)
vp_fl_sq = importlib.import_module(swi.vp_fluid_sq)
vp_M = importlib.import_module(swi.vp_M)
pr_bc = importlib.import_module(swi.prout_bc)

# sign


'''

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
    msh.transfer_2d_to_1d(fsp.v_square_n_1, fsp.v_square_n_1_0_1_on_1, os.path.join(rarg.args.input_directory, f'mesh_{0}'), rmsh.lmsh.parameters['shape_id'])

    
    vp_I = importlib.reload(vp_I)

    var_pr.solve_vp(vp_I.F_U, fsp.U_n_12, vp_I.bcs, fsp.J_U, parameters=params)
    
    print('... done.', flush=True)


    # step 2): solve D problem
    print('Solving D problem ...', flush=True)

    # now that U_n_12 has been computed, compute the new normal
    # POTENTIAL PROBLEM HERE: YOU MAY NEED TO USE A DISCRETE VERSION OF n_ale, using the relation between n and nu
    fsp.n_n_12.assign(project(bgeo.n_ale(fsp.ys, fsp.U_n_12), fsp.Q_U))

    # transfer v_square_n_1 and sigma_square_n_32 (defined on sub_mes[0][1]) on sub_mesh[0][0], and write the result in v_square_n_1_0_1_on_0_0 and sigma_square_n_32_0_1_on_0_0, respectively
    fsp.v_square_n_1_0_1_on_0_0.assign(project(fsp.v_square_n_1, fsp.Q_u_di_dot))
    fsp.sigma_square_n_32_0_1_on_0_0.assign(project(fsp.sigma_square_n_32, fsp.Q_sigma_disk))


    #transfer the new normal it from mesh[1] to sub_mesh[0][0]
    # POTENTIAL PROBLEM HERE: YOU MAY NEED TO USE A DISCRETE VERSION OF n_ale, using the relation between n and nu
    msh.transfer_1d_to_2d(fsp.n_n_12, fsp.n_n_12_1_on_0_0, os.path.join(rarg.args.input_directory, f'mesh_{0}'), rmsh.lmsh.parameters['shape_id'])

    fsp.u_n_di_dot_bc_di.assign(project(geo.euclidean_projection(fsp.v_square_n_1_0_1_on_0_0, fsp.n_n_12_1_on_0_0), fsp.Q_u_di_dot))

    #transfer the new normal it from mesh[1] to sub_mesh[0][1]
    # POTENTIAL PROBLEM HERE: YOU MAY NEED TO USE A DISCRETE VERSION OF n_ale, using the relation between n and nu
    msh.transfer_1d_to_2d(fsp.n_n_12, fsp.n_n_12_1_on_0_1,  os.path.join(rarg.args.input_directory, f'mesh_{0}'), rmsh.lmsh.parameters['shape_id'])

    fsp.u_n_sq_dot_bc_di.assign(project(geo.euclidean_projection(fsp.v_square_n_1, fsp.n_n_12_1_on_0_1), fsp.Q_u_sq_dot))

    # now that U_n_12 has been computed, transfer it from mesh[1] to sub_mesh[0][1] and from mesh[1] to sub_mesh[0][0]
    msh.transfer_1d_to_2d(fsp.U_n_12, fsp.U_n_12_1_on_0_1,  os.path.join(rarg.args.input_directory, f'mesh_{0}'), rmsh.lmsh.parameters['shape_id'])
    msh.transfer_1d_to_2d(fsp.U_n_12, fsp.U_n_12_1_on_0_0,  os.path.join(rarg.args.input_directory, f'mesh_{0}'), rmsh.lmsh.parameters['shape_id'])

    vp_D = importlib.reload(vp_D)

    # 2.1) solve for D in square

    var_pr.solve_vp(vp_D.F_u_sq, fsp.u_n_sq, vp_D.bcs_u_sq, fsp.J_u_sq, parameters=params)
    var_pr.solve_vp(vp_D.F_u_sq_dot, fsp.u_n_sq_dot, vp_D.bcs_u_sq_dot, fsp.J_u_dot_sq, parameters=params)

    # 2.2) solve for D in disk

    var_pr.solve_vp(vp_D.F_u_di, fsp.u_n_di, vp_D.bcs_u_di, fsp.J_u_di, parameters=params)
    var_pr.solve_vp(vp_D.F_u_di_dot, fsp.u_n_di_dot, vp_D.bcs_u_di_dot, fsp.J_u_dot_di, parameters=params)

    print('... done.', flush=True)


    # 3) solve for disk fluid 

    print('Solving disk fluid problem ...', flush=True)

    vp_fl_di = importlib.reload(vp_fl_di)

    # 3.1 solve for v_disk__

    var_pr.solve_vp(vp_fl_di.F_v_disk__, fsp.v_disk__, vp_fl_di.bc_v_disk__, fsp.J_v__disk, parameters=params)

    # 3.2 solve for phi_disk (and omega_disk)

    var_pr.solve_vp(vp_fl_di.F_phi_omega_disk, fsp.phi_omega_disk, vp_fl_di.bc_phi_omega_disk, fsp.J_phi_omega_disk, parameters=params)

    # 3.3 solve for v_disk_n

    var_pr.solve_vp(vp_fl_di.F_v_disk_n, fsp.v_disk_n, vp_fl_di.bc_v_disk_n, fsp.J_v_disk, parameters=params)


    # transfer v_disk_n (defined on sub_mesh[0][0]) on sub_mesh[0][1], and write the result in v_disk_n_0_0_on_0_1
    fsp.v_disk_n_0_0_on_0_1.assign(project(fsp.v_disk_n, fsp.Q_v__square))

    print('... done.', flush=True)


    # 4) solve for square fluid 

    print('Solving square fluid problem ...', flush=True)


    vp_fl_sq = importlib.reload(vp_fl_sq)

    # 4.1 solve for v_square__

    var_pr.solve_vp(vp_fl_sq.F_v_square__, fsp.v_square__, vp_fl_sq.bc_v_square__, fsp.J_v__square, parameters=params)

    # 4.2 solve for phi_square

    var_pr.solve_vp(vp_fl_sq.F_phi_square, fsp.phi_square, vp_fl_sq.bc_phi_square, fsp.J_phi_square, parameters=params)

    # 4.3 solve for v_square_n

    var_pr.solve_vp(vp_fl_sq.F_v_square_n, fsp.v_square_n, vp_fl_sq.bc_v_square_n, fsp.J_v_square, parameters=params)

    print('... done.', flush=True)



    # 5) solve for M

    print('Solving M problem ...', flush=True)

    vp_M = importlib.reload(vp_M)

    # solve for c_n

    var_pr.solve_vp(vp_M.F_c, fsp.c_n, vp_M.bc_M, fsp.J_c, parameters=params)
    
    print('... done.', flush=True)

    # print out the residuals of BCs
    # note: print_bcs() must be before the fields update to print the correct residuals of BCs
    if step % rpam.parameters['print_out_stride'] == 0:
        pr_bc.print_bcs()



    # update the fields
    # 1) I 

    fsp.U_n_32.assign(fsp.U_n_12)

    # 2) D

    # 2.1) disk
    fsp.u_n_2_di.assign(fsp.u_n_1_di)
    fsp.u_n_1_di.assign(fsp.u_n_di)

    fsp.u_n_2_di_dot.assign(fsp.u_n_1_di_dot)
    fsp.u_n_1_di_dot.assign(fsp.u_n_di_dot)

    # 2.2) square
    fsp.u_n_2_sq.assign(fsp.u_n_1_sq)
    fsp.u_n_1_sq.assign(fsp.u_n_sq)

    fsp.u_n_2_sq_dot.assign(fsp.u_n_1_sq_dot)
    fsp.u_n_1_sq_dot.assign(fsp.u_n_sq_dot)


    # 3) disk fluid 

    phi_disk_output, omega_disk_output = fsp.phi_omega_disk.split(deepcopy=True)
    fsp.phi_disk_on_Q_sigma_disk.interpolate(phi_disk_output)
    fsp.sigma_disk_n_12.assign(fsp.sigma_disk_n_32 - fsp.phi_disk_on_Q_sigma_disk)

    fsp.v_disk_n_2.assign(fsp.v_disk_n_1)
    fsp.v_disk_n_1.assign(fsp.v_disk_n)

    fsp.sigma_disk_n_32.assign(fsp.sigma_disk_n_12)


    # 4) square fluid 

    fsp.sigma_square_n_12.assign(fsp.sigma_square_n_32 - fsp.phi_square)

    fsp.v_square_n_2.assign(fsp.v_square_n_1)
    fsp.v_square_n_1.assign(fsp.v_square_n)

    fsp.sigma_square_n_32.assign(fsp.sigma_square_n_12)

    # 5) M

    fsp.c_n_1.assign(fsp.c_n)



    # print out the solution
    if step % rpam.parameters['print_out_stride'] == 0:
        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        pr_sol.print_solution(t, step)


    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)
'''