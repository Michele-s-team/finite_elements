"""
This code solves for the dynamics of the Navier Stokes equations on a fixed, curved manifold with Crank Nicholson discretization scheme

Run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution] T N

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_curved_cn/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_curved_cn/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $SOLUTION_PATH

    


To reproduce air flow with  obstacle (figure 9):
-        - select  variational_problem_bc_obstacle
-        - set L = 2.2
-        - set c_r = [0.2, h/2, 0]
-        - set outflow = 'near(x[0], 2.2)
-        - set c_r = [0.2, h/2]
-        - set cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.05 && x[1]<0.41-0.05'
-        - set
-                * rho = 1.293e-2
-                * mu = 1.85e-7
-                * v0 = 1e-4
-                * v_l = 1e2 * v0
-        - set v__profile_l = Expression(('4.0*1.5*x[1]*(0.41 - x[1]) / pow(h, 2) * v_l', '0'), degree=2, v_l=v_l, h=rmsh.h)
-        - set
-                class ManifoldExpression( UserExpression ):
-                    def eval(self, values, x):
-                        values[0] = 2 * x[1] * (rmsh.h - x[1]) / rmsh.h**2 * (x[1] - rmsh.h / 24) / rmsh.h
-                    def value_shape(self):
-                        return (1,)
-
-                class OmegaExpression( UserExpression ):
-                    def eval(self, values, x):
-                        values[0] = 0
-                        values[1] = -((rmsh.h**2) - 50.0*rmsh.h*x[1] + 72.0*((x[1])**2))/(12.0*rmsh.h**3)
-                    def value_shape(self):
-                        return (2,)
-        - run with
-            * clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_mesh.py 0.05 $SOLUTION_PATH
-            * clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir -p /home/fenics/shared/dynamics/channel_with_cylinder_curved_cn/$SOLUTION_PATH/snapshots/csv/nodal_values; python3 solve.py /home/fenics/shared/dynamics/channel_with_cylinder_curved_cn/mesh/solution /home/fenics/shared/dynamics/channel_with_cylinder_curved_cn/$SOLUTION_PATH  128 128
"""

import dolfin
from fenics import *
import importlib
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi
import print_out_solution as pr_sol
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
pr_bc = importlib.import_module(swi.prout_bc)
pr_da = importlib.import_module(swi.prout_da)

dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

# 1.1 store mesh metadata
mesh_metadata = rmsh.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'mesh_metadata.csv'), mesh_metadata)

# 1.2 store solution metadata
solution_metadata = rpam.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'solution_metadata.csv'), solution_metadata)



# set the initial profiles
fsp.v_n_1.interpolate(vp.TangentVelocityExpression(element=fsp.Q_v.ufl_element()))
fsp.v_n_2.assign(fsp.v_n_1)
fsp.w.interpolate(vp.NormalVelocityExpression(element=fsp.Q.ufl_element()))
fsp.sigma_n_12.interpolate(vp.SurfaceTensionExpression(element=fsp.Q.ufl_element()))
fsp.sigma_n_32.assign(fsp.sigma_n_12)
fsp.z.interpolate(vp.ManifoldExpression(element=fsp.Q_z.ufl_element()))
fsp.omega.interpolate(vp.OmegaExpression(element=fsp.Q_omega.ufl_element()))

pr_sol.print_z_omega()

params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                'linear_solver': 'default',
                'absolute_tolerance': 1e-10,
                'relative_tolerance': 1e-9,
                'maximum_iterations': 50,
                'relaxation_parameter': None,
                'preconditioner': 'default'
              }
          }

print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters['num_steps']):

    # Update current time
    t += vp.dt
    step += 1

    vp = importlib.import_module(swi.vp)

    # step 1: tentative velocity step
    var_pr.solve_vp(vp.F1, fsp.v_, vp.bc_v_, fsp.J_v_, parameters=params)

    # step 2: surface_tension correction step
    var_pr.solve_vp(vp.F2, fsp.phi, vp.bc_phi, fsp.J_phi, parameters=params)

    # step 3: velocity step
    var_pr.solve_vp(vp.F3, fsp.v_n, vp.bc_v_n, fsp.J_v_n, parameters=params)


    pr_bc.print_bcs()
    pr_da.print_data(step)

    # obtain fsp.sigma_n from fsp.phi by using the definition of fsp.phi
    fsp.sigma_n_12.assign(fsp.sigma_n_32 - fsp.phi)

    # Update previous solution
    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(fsp.v_n)

    fsp.sigma_n_32.assign(fsp.sigma_n_12)

    pr_sol.print_solution(t, step, vp.dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)
