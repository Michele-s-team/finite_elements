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

import differential_geometry.manifold.geometry as geo
import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import print_out_solution as pr_sol
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
vp_D = importlib.import_module(swi.vp_D)
vp_fl_di = importlib.import_module(swi.vp_fluid_di)
vp_fl_sq = importlib.import_module(swi.vp_fluid_sq)

io.full_print(fsp.ys, 'ys', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path, \
              lmsh.mesh[1], 'vector')

# FILL IN HWERE: set the initial profiles from analytical expressions
# REMEMBER TO TRANSFER FUNCTIONS DURING TIME ITERATION


'''
# set an initial value for v_square_n_1
class v_sq_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 1

    def value_shape(self):
        return (2,)
fsp.v_square_n_1.interpolate(v_sq_expression(element=fsp.Q_v_square.ufl_element()))
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
    msh.transfer_circle_to_line(fsp.v_square_n_1, fsp.v_square_n_1_0_1_on_1, lmsh.mesh_parameters[0]['c_r'], lmsh.mesh_parameters[0]['r'], lmsh.mesh_parameters[0]['N'])
    
    vp_I = importlib.reload(vp_I)

    J_I = derivative(vp_I.F_U, fsp.U_n_12, fsp.J_U)
    problem_I = NonlinearVariationalProblem(vp_I.F_U, fsp.U_n_12, vp_I.bcs, J_I)
    solver_I = NonlinearVariationalSolver(problem_I)
    solver_I.parameters.update(params)
    solver_I.solve()

    print('... done.', flush=True)


    # step 2): solve D problem
    print('Solving D problem ...', flush=True)

    # now that U_n_12 has been computed, compute the new normal
    fsp.n_n_12.assign(project(bgeo.n_ale(fsp.ys, fsp.U_n_12), fsp.Q_U))

    # transfer v_square_n_1 and sigma_square_n_32 (defined on sub_mes[0][1]) on sub_mesh[0][0], and write the result in v_square_n_1_0_1_on_0_0 and sigma_square_n_32_0_1_on_0_0, respectively
    fsp.v_square_n_1_0_1_on_0_0.assign(project(fsp.v_square_n_1, fsp.Q_u_di_dot))
    fsp.sigma_square_n_32_0_1_on_0_0.assign(project(fsp.sigma_square_n_32, fsp.Q_sigma_disk))


    #transfer the new normal it from mesh[1] to sub_mesh[0][0]
    msh.transfer_line_to_circle(fsp.n_n_12, fsp.n_n_12_1_on_0_0, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])
    fsp.u_n_di_dot_bc_di.assign(project(geo.euclidean_projection(fsp.v_square_n_1_0_1_on_0_0, fsp.n_n_12_1_on_0_0), fsp.Q_u_di_dot))

    #transfer the new normal it from mesh[1] to sub_mesh[0][1]
    msh.transfer_line_to_circle(fsp.n_n_12, fsp.n_n_12_1_on_0_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])
    fsp.u_n_sq_dot_bc_di.assign(project(geo.euclidean_projection(fsp.v_square_n_1, fsp.n_n_12_1_on_0_1), fsp.Q_u_sq_dot))

    # now that U_n_12 has been computed, transfer it from mesh[1] to sub_mesh[0][1] and from mesh[1] to sub_mesh[0][0]
    msh.transfer_line_to_circle(fsp.U_n_12, fsp.U_n_12_1_on_0_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])
    msh.transfer_line_to_circle(fsp.U_n_12, fsp.U_n_12_1_on_0_0, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])

    vp_D = importlib.reload(vp_D)

    # 2.1) solve for D in square

    J_u_sq = derivative(vp_D.F_u_sq, fsp.u_n_sq, fsp.J_u_sq)
    problem_u_sq = NonlinearVariationalProblem(vp_D.F_u_sq, fsp.u_n_sq, vp_D.bcs_u_sq, J_u_sq)
    solver_u_sq = NonlinearVariationalSolver(problem_u_sq)

    J_u_dot_sq = derivative(vp_D.F_u_sq_dot, fsp.u_n_sq_dot, fsp.J_u_dot_sq)
    problem_u_dot_sq = NonlinearVariationalProblem(vp_D.F_u_sq_dot, fsp.u_n_sq_dot, vp_D.bcs_u_sq_dot, J_u_dot_sq)
    solver_u_dot_sq = NonlinearVariationalSolver(problem_u_dot_sq)

    solver_u_sq.parameters.update(params)
    solver_u_dot_sq.parameters.update(params)

    solver_u_sq.solve()
    solver_u_dot_sq.solve()

    # 2.2) solve for D in disk
    J_u_di = derivative(vp_D.F_u_di, fsp.u_n_di, fsp.J_u_di)
    problem_u_di = NonlinearVariationalProblem(vp_D.F_u_di, fsp.u_n_di, vp_D.bcs_u_di, J_u_di)
    solver_u_di = NonlinearVariationalSolver(problem_u_di)

    J_u_dot_di = derivative(vp_D.F_u_di_dot, fsp.u_n_di_dot, fsp.J_u_dot_di)
    problem_u_dot_di = NonlinearVariationalProblem(vp_D.F_u_di_dot, fsp.u_n_di_dot, vp_D.bcs_u_di_dot, J_u_dot_di)
    solver_u_dot_di = NonlinearVariationalSolver(problem_u_dot_di)

    solver_u_di.parameters.update(params)
    solver_u_dot_di.parameters.update(params)

    solver_u_di.solve()
    solver_u_dot_di.solve()

    print('... done.', flush=True)

    # 3) solve for disk fluid 

    print('Solving disk fluid problem ...', flush=True)

    vp_fl_di = importlib.reload(vp_fl_di)

    # 3.1 solve for v_disk__
    J_fl_di_v__ = derivative(vp_fl_di.F_v_disk__, fsp.v_disk__, fsp.J_v__disk)
    problem_fl_di_v__ = NonlinearVariationalProblem(vp_fl_di.F_v_disk__, fsp.v_disk__, vp_fl_di.bc_v_disk__, J_fl_di_v__)
    solver_fl_di_v__ = NonlinearVariationalSolver(problem_fl_di_v__)
    solver_fl_di_v__.solve()

    # 3.2 solve for phi_disk (and omega_disk)
    J_fl_di_phi_omega = derivative(vp_fl_di.F_phi_omega_disk, fsp.phi_omega_disk, fsp.J_phi_omega_disk)
    problem_fl_di_phi_omega = NonlinearVariationalProblem(vp_fl_di.F_phi_omega_disk, fsp.phi_omega_disk, vp_fl_di.bc_phi_omega_disk, J_fl_di_phi_omega)
    solver_fl_di_phi_omega = NonlinearVariationalSolver(problem_fl_di_phi_omega)
    solver_fl_di_phi_omega.solve()

    # 3.3 solve for v_disk_n
    J_fl_di_v_n = derivative(vp_fl_di.F_v_disk_n, fsp.v_disk_n, fsp.J_v_disk)
    problem_fl_di_v_n = NonlinearVariationalProblem(vp_fl_di.F_v_disk_n, fsp.v_disk_n, vp_fl_di.bc_v_disk_n, J_fl_di_v_n)
    solver_fl_di_v_n = NonlinearVariationalSolver(problem_fl_di_v_n)
    solver_fl_di_v_n.solve()

    # transfer v_disk_n (defined on sub_mesh[0][0]) on sub_mesh[0][1], and write the result in v_disk_n_0_0_on_0_1
    fsp.v_disk_n_0_0_on_0_1.assign(project(fsp.v_disk_n, fsp.Q_v__square))

    print('... done.', flush=True)

    # sign

    # 4) solve for square fluid 

    print('Solving square fluid problem ...', flush=True)


    print('... done.', flush=True)

    pr_sol.print_solution(t, step)







print("... done.", flush=True)
