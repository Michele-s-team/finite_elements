"""
This code solves for the dynamics of the Navier Stokes equations with an elastic obstacle which is pinned on part of its boundary
 on a flat manifold Crank Nicholson discretization scheme

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse_circle/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_ellipse_circle $MESH_PATH $SOLUTION_PATH
"""

import dolfin
from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import physics.fluid_mechanics as flu
import function_spaces as fsp
import input_output as io
import print_out_solution as pr_sol
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi
import variational_problem.utils as var_pr


dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

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

pr_bc = importlib.import_module(swi.prout_bc)
rmsh = importlib.import_module(swi.rmsh)
vp_el = importlib.import_module(swi.vp_el)
vp_fl = importlib.import_module(swi.vp_fl)
vp_msh = importlib.import_module(swi.vp_msh)
'''
pr_bc = importlib.import_module(swi.prout_bc)
'''

dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

io.full_print(fsp.ys_ellipse, 'ys_ellipse', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path)

io.full_print(fsp.dyds_ellipse, 'dyds_ellipse', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path)

# fork:
# a) read initial profiles from file
'''
step_ic = 100
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_el_n_{step_ic-1}.h5', "r").read(fsp.u_el_n_1, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_el_n_{step_ic-2}.h5', "r").read(fsp.u_el_n_2, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_el_dot_n_{step_ic-1}.h5', "r").read(fsp.u_el_dot_n_1, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_el_dot_n_{step_ic-2}.h5', "r").read(fsp.u_el_dot_n_2, "/f")

HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_msh_n_{step_ic}.h5', "r").read(fsp.u_msh_n, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_msh_n_{step_ic-1}.h5', "r").read(fsp.u_msh_n_1, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_msh_n_{step_ic-2}.h5', "r").read(fsp.u_msh_n_2, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_msh_dot_n_{step_ic}.h5', "r").read(fsp.u_msh_dot_n, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_msh_dot_n_{step_ic-1}.h5', "r").read(fsp.u_msh_dot_n_1, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/u_msh_dot_n_{step_ic-2}.h5', "r").read(fsp.u_msh_dot_n_2, "/f")
fsp.u_msh_dot_n_2.assign(fsp.u_msh_dot_n_1)


HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/v_n_{step_ic}.h5', "r").read(fsp.v_n, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/v_n_{step_ic-1}.h5', "r").read(fsp.v_n_1, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/v_n_{step_ic-2}.h5', "r").read(fsp.v_n_2, "/f")

HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/sigma_n_12_{step_ic}.h5', "r").read(fsp.sigma_n_12, "/f")
HDF5File(MPI.comm_world, f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/solution_ic/snapshots/h5/sigma_n_12_{step_ic-1}.h5', "r").read(fsp.sigma_n_32, "/f")
'''


# b) set the initial profiles from analytical expressions
#
fsp.v_n_1.interpolate(vp_fl.v_expression(element=fsp.Q_v.ufl_element()))
fsp.v_n_2.assign(fsp.v_n_1)
fsp.sigma_n_12.interpolate(vp_fl.sigma_expression(element=fsp.Q_phi.ufl_element()))
fsp.sigma_n_32.assign(fsp.sigma_n_12)
# 

print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters['num_steps']):
    # Update current time
    t += dt
    step += 1


    # step 1): solve elastic problem
    print('Solving elastic problem ...', flush=True)


    # project v_n_1 and sigma_n_32 on the mesh of the elastic problem to define the BC for the elastic problem
    fsp.v_n_1_on_sub_mesh_0.assign(project(fsp.v_n_1, fsp.Q_v_el))
    fsp.sigma_n_32_on_sub_mesh_0.assign(project(fsp.sigma_n_32, fsp.Q_sigma_el))

    # print out force exerted by  fl on el
    #
    import elasticity as ela
    import ufl
    import differential_geometry.manifold.geometry as geo
    import input_output as io
    import solution_paths as solpath
    import mesh.load as lmsh

    i, j, k, l, m = ufl.indices(5)
    f = Function(fsp.Q_u_el)
    f.assign(project(as_tensor((flu.sigma_ale(fsp.v_n_1_on_sub_mesh_0, fsp.sigma_n_32_on_sub_mesh_0, fsp.u_el_n, rpam.parameters['mu_fluid'])[i, j] * geo.epsilon[j, k] * ela.F(fsp.u_el_n_1)[k, l] * fsp.dyds_ellipse[l] / sqrt(fsp.dyds_ellipse[m] * fsp.dyds_ellipse[m])), (i)), fsp.Q_u_el))
    io.full_print(f, 'F_el_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path)
    #

    vp_el = importlib.reload(vp_el)

    var_pr.solve_vp(vp_el.F_el, fsp.psi_el, vp_el.bcs_el, fsp.J_psi_el, parameters=params)

    # pr_sol.print_solution_el(t, step)

    print('... done.', flush=True)


    # step 2): update u and u_dot (mesh problem)
    print('Solving mesh problem ...', flush=True)

    # define fields from mesh 0 on mesh 1 form the elastic problem to define BCs for the mesh problem
    u_el_n_output, u_el_dot_n_output = fsp.psi_el.split(deepcopy=True)
    u_el_n_output.set_allow_extrapolation(True)
    u_el_dot_n_output.set_allow_extrapolation(True)
    fsp.u_el_n_on_sub_mesh_1.assign(project(u_el_n_output, fsp.Q_u_msh))
    fsp.u_el_dot_n_on_sub_mesh_1.assign(project(u_el_dot_n_output, fsp.Q_u_msh_dot))

    vp_msh = importlib.reload(vp_msh)

    var_pr.solve_vp(vp_msh.F_msh_u, fsp.u_msh_n, vp_msh.bcs_msh, fsp.J_msh_u, parameters=params)
    var_pr.solve_vp(vp_msh.F_msh_u_dot, fsp.u_msh_dot_n, vp_msh.bcs_msh_dot, fsp.J_msh_u_dot, parameters=params)

    print('... done.', flush=True)

    # step 3) update v_n and sigma_n_12 (fluid problem)
    print('Solving fluid problem ...', flush=True)

    vp_fl = importlib.reload(vp_fl)

    # step 3.1
    var_pr.solve_vp(vp_fl.F_v_, fsp.v_, vp_fl.bc_v_, fsp.J_v_)

    # Step 3.2: surface_tension correction step
    var_pr.solve_vp(vp_fl.F_phi, fsp.phi, vp_fl.bc_phi, fsp.J_phi)

    # step 3.3
    var_pr.solve_vp(vp_fl.F_v_n, fsp.v_n, [], fsp.J_v_n)

    print('... done.', flush=True)

    # note: print_bcs() must be before the fields update to print the correct residuals of BCs
    pr_bc.print_bcs()


    # update the fields
    # 1) update the elastic problem
    u_el_n_output, u_el_dot_n_output = fsp.psi_el.split( deepcopy=True)

    fsp.u_el_n_2.assign(fsp.u_el_n_1)
    fsp.u_el_n_1.assign(u_el_n_output)

    fsp.u_el_dot_n_2.assign(fsp.u_el_dot_n_1)
    fsp.u_el_dot_n_1.assign(u_el_dot_n_output)


    # 2) update the mesh problem
    fsp.u_msh_n_2.assign(fsp.u_msh_n_1)
    fsp.u_msh_n_1.assign(fsp.u_msh_n)

    fsp.u_msh_dot_n_2.assign(fsp.u_msh_dot_n_1)
    fsp.u_msh_dot_n_1.assign(fsp.u_msh_dot_n)


    # 3) update the fluid problem
    fsp.sigma_n_12.assign(fsp.sigma_n_32 - fsp.phi)

    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(fsp.v_n)

    fsp.sigma_n_32.assign(fsp.sigma_n_12)
    
    if step % rpam.parameters['print_out_stride'] == 0:
        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        pr_sol.print_solution(t, step, dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)

