"""
This code solves for the dynamics of the Navier Stokes equations for a fluid in a square whose top edge is a membrane. The coupled dynamics of  membrane, fluid and of the fictitious elastic body (which defines the region where the fluid moves) are solved. 

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/membrane/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_line_a $MESH_PATH $SOLUTION_PATH
"""
import colorama as col
import dolfin
from fenics import *
import gc
import importlib
import numpy as np
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
import physics.fluid_mechanics as flu
import function as fu
import input_output as io
import mesh.utils as msh
import mesh_quality as msh_qu
import parameters.read.solution as rpam
import physics.utils as phys
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr

fi = importlib.import_module(swi.fi)

'''
# test transfer sub mesh to sub mesh - start
import function as fu
import solution_paths as solpath


# read the mesh
path_a = '/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution_a'
path_b = '/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution_b'

parameters_a = io.read_parameters_from_csv_file(os.path.join(path_a, "mesh_metadata.csv"))
parameters_b = io.read_parameters_from_csv_file(os.path.join(path_b, "mesh_metadata.csv"))


mesh_a, sf_a = msh.read_from_file(path_a, 'xdmf')
mesh_b, sf_b = msh.read_from_file(path_b, 'xdmf')

print(f'number of vertices = {mesh_a.num_vertices()} {mesh_b.num_vertices()}')


# read the sub_meshes and generate their functions tagging cells and vertices
sub_meshes_a, sf_sub_meshes_a, mf_sub_meshes_a = msh.read_sub_meshes(mesh_a, sf_a, parameters_a, path_a)
sub_meshes_b, sf_sub_meshes_b, mf_sub_meshes_b = msh.read_sub_meshes(mesh_b, sf_b, parameters_b, path_b)

class u_a_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]**2
        values[1] = x[0]**3
        values[2] = x[0]**2-1
        values[3] = x[0]**2-2


    def value_shape(self):
        return (4,)
    
class u_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = - x[0] * (x[0]-1)


    def value_shape(self):
        return (2,)

Q_u_a = TensorFunctionSpace(sub_meshes_a[1], 'P', 2, shape=(2,2))
Q_u_b = TensorFunctionSpace(sub_meshes_b[1], 'P', 2, shape=(2,2))

Q_u = VectorFunctionSpace(sub_meshes_a[0], 'P', 2)

u = Function(Q_u)
u_a = Function(Q_u_a)
u_b = Function(Q_u_b)

u.interpolate(u_expression(element=Q_u.ufl_element()))
u_a.interpolate(u_a_expression(element=Q_u_a.ufl_element()))

fu.transfer_sub_mesh_to_sub_mesh(u_a, u_b, u, path_a)

io.full_print(u, 'u_test', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
io.full_print(u_a, 'u_a_test', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
io.full_print(u_b, 'u_b_test', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)


# test transfer sub mesh to sub mesh - end

'''

'''
# test transfer_mesh_to_sub_mesh - start 
import function as fu
import input_output as io
import mesh.load as lmsh
import runtime_arguments as rarg
import solution_paths as solpath



class u_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]**2-x[1]
        values[1] = x[0]**3-x[1]
        values[2] = x[0]**4-x[1]
        values[3] = x[0]**5-x[1]

    def value_shape(self):
        return (2,2)

Q_0 = TensorFunctionSpace(lmsh.sub_meshes[0], 'P', 2, shape=(2,2))
Q_1 = TensorFunctionSpace(lmsh.sub_meshes[1], 'P', 2, shape=(2,2))

u_0 = Function(Q_0)
u_1 = Function(Q_1)

u_0.interpolate(u_0_expression(element=Q_0.ufl_element()))


fu.transfer_mesh_to_sub_mesh(u_0, u_1, rarg.args.input_directory)

io.full_print(u_0, 'u_0_test', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
io.full_print(u_1, 'u_1_test', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

# test transfer_mesh_to_sub_mesh - end
'''

mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 
pre_remesh_path = os.path.join(rarg.args.input_directory, '../solution_pre_remesh')
os.system(f'rm -rf {pre_remesh_path}')

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size

# set the solver parameters here
# parameters with Netwon method
'''
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


print(f'Generating initial mesh ...')
# generate the mesh with the curve given by curve_coordinates and write into its mesh_metadata

msh.generate_square_no_circle_curve_mesh(mesh_parameters['curve_coordinates'], os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)

print(f'... done.')

# first load of modules
import differential_geometry.manifold.geometry as geo
import differential_geometry.boundary.geometry as bgeo

import function_spaces as fsp

pr_bc = importlib.import_module(swi.prout_bc)
pr_da = importlib.import_module(swi.prout_da)
pr_sol = importlib.import_module(swi.prout_sol)
rmsh = importlib.import_module(swi.rmsh)
cu = importlib.import_module(swi.cu)


# test calls of problems

# 1. membrane problem
fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rarg.args.input_directory)

vp_membrane = importlib.import_module(swi.vp_membrane)

fsp.sigma_n_32.interpolate(vp_membrane.sigma_n_32_0_Expression(element=fsp.Q_psi_n_12.ufl_element()))


# 2. mesh problem
# project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
# a) project U_n_12
v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )

fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh, rarg.args.input_directory)
# b) project U_dot_n_12
fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh, rarg.args.input_directory)

vp_mesh = importlib.import_module(swi.vp_mesh)

# 3. fluid problem
vp_fluid = importlib.import_module(swi.vp_fluid)

dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']


# 1. store metadata

# 1.1 store mesh metadata
mesh_metadata = rmsh.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'mesh_metadata.csv'), mesh_metadata)

# 1.2 store solution metadata
solution_metadata = rpam.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'solution_metadata.csv'), solution_metadata)


#2. set the initial profiles


#2.1 set from expressions

# 2.1.1 for the membrane
fsp.v_bar_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_bar.ufl_element() ) )
fsp.v_n_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_n.ufl_element() ) )
fsp.nu_n_12_0.interpolate( vp_membrane.nu_n_12_0_Expression( element=fsp.Q_nu_n_12.ufl_element() ) )
fsp.U_n_12_0.interpolate( vp_membrane.U_n_12_0_Expression( element=fsp.Q_U_n_12.ufl_element() ) )
# 2.1.2 for the mesh
# 2.1.3 for the fluid
# fsp.v_n_1.interpolate(vp_fl.v_expression(element=fsp.Q_v.ufl_element()))
# fsp.v_n_2.assign(fsp.v_n_1)
fsp.sigma_fl_n_12.interpolate(vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

fsp.assigner_mem.assign(fsp.psi_mem, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0, fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0 ])


'''
#2.2 read initial profiles by reading them from file
'''

#3. Time-stepping

print("Starting time iteration ...", flush=True)

t = 0
step = 0

for n in range(rpam.parameters['N']):

    #3.1 update current time

    t += dt
    step += 1

    #3.2 solve variational problems
    
    #3.2.1 solve membrane problem 
    print('Solving membrane problem ...', flush=True)
   
    # project from sub_mesh[0] onto sub_mesh[1] the fields from the fluid problem, in order to find the force exerted by the fluid on the membrane 
    fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
    fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rarg.args.input_directory)
    
    vp_membrane = importlib.reload(importlib.import_module(swi.vp_membrane))  

    var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem, vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)

    print('... done.', flush=True)


    #3.2.2 solve mesh problem

    print('Solving mesh problem ...', flush=True)
    
    # project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
    # a) project U_n_12
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
    fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh, rarg.args.input_directory)
    # b) project U_dot_n_12
    fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
    fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh, rarg.args.input_directory)

    vp_mesh = importlib.reload(importlib.import_module(swi.vp_mesh))  

    # solve for u_n and u_dot_n
    var_pr.solve_vp(vp_mesh.F_msh, fsp.u_n, vp_mesh.bcs_msh, fsp.J_u, parameters=params)
    var_pr.solve_vp(vp_mesh.F_msh_dot, fsp.u_dot_n, vp_mesh.bcs_msh_dot, fsp.J_u_dot, parameters=params)

    print('... done.', flush=True)


    # 3.2.3 solve fluid problem

    print('Solving fluid problem ...', flush=True)

    vp_fluid = importlib.reload(importlib.import_module(swi.vp_fluid))  

    # step 3.2.3.1: approximate velocity step
    var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar, vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)

    # Step 3.2.3.2: surface_tension correction step
    var_pr.solve_vp(vp_fluid.F_phi_fl, fsp.phi_fl, vp_fluid.bc_phi_fl, fsp.J_phi_fl, parameters=params)

    # step 3.2.3.3: velocity 
    var_pr.solve_vp(vp_fluid.F_v_fl_n, fsp.v_fl_n, [], fsp.J_v_fl_n, parameters=params)

    print('... done.', flush=True)
    

    #3.3 print BCs, ICs, data such as mesh quality. Note: print_bcs and print_ics must be before the fields update to print the correct residuals of BCs

    #3.3.1 compute mesh quality
    msh_qu.quality = msh.custom_mesh_quality(msh.deform_mesh(rmsh.lmsh.sub_meshes[0], fsp.u_n))


    #3.3.3 compure BCs and data
    pr_bc.print_bcs(step)
    pr_da.print_data(step)


    if msh_qu.quality < rpam.parameters['mesh_quality_threshold']:
    # if True:

        #4. remesh (the mesh quality got below mesh_quality_threshold)

        print(f'{col.Fore.CYAN}Remeshing ... {col.Style.RESET_ALL}')

        #4.1 transfer fields

        #4.1.1 Define _old fields that store the last configurations from the last iteration with the previous mesh

        # 4.1.1.1 _old fields for membrane
        v_bar_old = Function(fsp.Q_v_bar)
        w_bar_old = Function(fsp.Q_w_bar)
        phi_old = Function(fsp.Q_phi)
        v_n_old = Function(fsp.Q_v_n)
        w_n_old = Function(fsp.Q_w_n)
        U_n_12_old = Function(fsp.Q_U_n_12)
        nu_n_12_old = Function(fsp.Q_nu_n_12)
        psi_n_12_old = Function(fsp.Q_psi_n_12)
        mu_n_12_old = Function(fsp.Q_mu_n_12)

        v_n_1_old = Function(fsp.Q_v_n)
        v_n_2_old = Function(fsp.Q_v_n)
        w_n_1_old = Function(fsp.Q_w_n)
        sigma_n_12_old = Function(fsp.Q_phi)
        sigma_n_32_old = Function(fsp.Q_phi)
        U_n_32_old = Function(fsp.Q_U_n_12)


        # 4.1.1.2 _old for mesh
        u_n_old = Function(fsp.Q_u)
        u_n_1_old = Function(fsp.Q_u)
        u_n_2_old = Function(fsp.Q_u)

        u_dot_n_old = Function(fsp.Q_u_dot)
        u_dot_n_1_old = Function(fsp.Q_u_dot)
        u_dot_n_2_old = Function(fsp.Q_u_dot)

        # 4.1.1.3 _old fields for fluid
        v_fl_n_old = Function(fsp.Q_v_fl)
        v_fl_n_1_old = Function(fsp.Q_v_fl)
        v_fl_n_2_old = Function(fsp.Q_v_fl)

        phi_fl_old = Function(fsp.Q_phi_fl)

        sigma_fl_n_12_old = Function(fsp.Q_phi_fl)
        sigma_fl_n_32_old = Function(fsp.Q_phi_fl)


        #4.1.2 Write in the _old fields the configurations form the last iteration with the previous mesh

        #4.1.2.1 unpack the mixed field for membrane
        v_bar_dummy, w_bar_dummy, phi_dummy, v_n_dummy, w_n_dummy, U_n_12_dummy, nu_n_12_dummy, psi_n_12_dummy, mu_n_12_dummy = fsp.psi_mem.split(deepcopy=True)


        # 4.1.2.2 write into the _old fields

        # 4.1.2.2.1 write into membrane fields
        v_bar_old.assign(v_bar_dummy)
        w_bar_old.assign(w_bar_dummy)
        phi_old.assign(phi_dummy)
        v_n_old.assign(v_n_dummy)
        w_n_old.assign(w_n_dummy)
        U_n_12_old.assign(U_n_12_dummy)
        nu_n_12_old.assign(nu_n_12_dummy)
        psi_n_12_old.assign(psi_n_12_dummy)
        mu_n_12_old.assign(mu_n_12_dummy)

        v_n_1_old.assign(fsp.v_n_1)
        v_n_2_old.assign(fsp.v_n_2)
        w_n_1_old.assign(fsp.w_n_1)
        sigma_n_12_old.assign(fsp.sigma_n_12)
        sigma_n_32_old.assign(fsp.sigma_n_32)
        U_n_32_old.assign(fsp.U_n_32)

        # 4.1.2.2.1 write into mesh fields
        u_n_old.assign(fsp.u_n)
        u_n_1_old.assign(fsp.u_n_1)
        u_n_2_old.assign(fsp.u_n_2)

        u_dot_n_old.assign(fsp.u_dot_n)
        u_dot_n_1_old.assign(fsp.u_dot_n_1)
        u_dot_n_2_old.assign(fsp.u_dot_n_2)

        # 4.1.2.2.3 write into fluid fields
        v_fl_n_old.assign(fsp.v_fl_n)
        v_fl_n_1_old.assign(fsp.v_fl_n_1)
        v_fl_n_2_old.assign(fsp.v_fl_n_2)

        phi_fl_old.assign(fsp.phi_fl)

        sigma_fl_n_12_old.assign(fsp.sigma_fl_n_12)
        sigma_fl_n_32_old.assign(fsp.sigma_fl_n_32)


        # 4.2 build the new mesh 

        mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, 'mesh_metadata.csv')) 

        curve_coordinates = []
        for i in range(len(mesh_parameters["curve_coordinates"])):
            # run through all coordinates of the nodes of the boundary

            coordinate = mesh_parameters["curve_coordinates"][i]

            # the new reference coordinate is obtained by adding to the previous reference coordinate, the displacement field u_n
              
            curve_coordinates.append(np.add(coordinate, fsp.u_n(coordinate).tolist()).tolist())

        #4.2.1 generate the mesh with the new curve_coordinates
        
        # store the mesh before remeshing in `pre_remesh_path`, this will be needed for transferring fields
        os.system(f'rm -rf {pre_remesh_path}; mkdir -p {pre_remesh_path}; cp -r {rarg.args.input_directory}/. {pre_remesh_path}')

        msh.generate_square_no_circle_curve_mesh(curve_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)


        #4.3 reload modules so everything is updated according to the mesh change
        
        # ----- WARNING : FROM THIS LINE ON, FIELDS RELATIVE TO THE OLD MESH SET UP WILL BE OVERWRITTEN -----
        importlib.reload(geo)
        importlib.reload(rmsh.lmsh)
        importlib.reload(bgeo)
        importlib.reload(fsp)
        rmsh = importlib.reload(rmsh)
        pr_bc = importlib.reload(pr_bc)
        pr_da = importlib.reload(pr_da)
        pr_sol = importlib.reload(pr_sol)
        cu = importlib.reload(cu)



        #4.4 transfer the values stored in the _old fields to the fields defined on the new mesh

        # 4.4.1 transfer membrane fields
        fu.transfer_sub_mesh_to_sub_mesh(v_bar_old, fsp.v_bar_output, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(w_bar_old, fsp.w_bar_output, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(phi_old, fsp.phi_output, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(v_n_old, fsp.v_n_output, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(w_n_old, fsp.w_n_output, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(U_n_12_old, fsp.U_n_12_output, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(nu_n_12_old, fsp.nu_n_12_output, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(psi_n_12_old, fsp.psi_n_12_output, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(mu_n_12_old, fsp.mu_n_12_output, u_n_old, pre_remesh_path)

        fsp.assigner_mem.assign(fsp.psi_mem, [fsp.v_bar_output, fsp.w_bar_output, fsp.phi_output, fsp.v_n_output, fsp.w_n_output, fsp.U_n_12_output, fsp.nu_n_12_output, fsp.psi_n_12_output, fsp.mu_n_12_output])


        fu.transfer_sub_mesh_to_sub_mesh(v_n_1_old, fsp.v_n_1, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(v_n_2_old, fsp.v_n_2, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(w_n_1_old, fsp.w_n_1, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(sigma_n_12_old, fsp.sigma_n_12, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(sigma_n_32_old, fsp.sigma_n_32, u_n_old, pre_remesh_path)
        fu.transfer_sub_mesh_to_sub_mesh(U_n_32_old, fsp.U_n_32, u_n_old, pre_remesh_path)



        # 4.4.2 set mesh fields
        fsp.u_n.assign(Constant((0, 0)))
        fsp.u_n_1.assign(Constant((0, 0)))
        fsp.u_n_2.assign(Constant((0, 0)))

        msh.transfer(u_dot_n_old, fsp.u_dot_n, u_n_old)
        msh.transfer(u_dot_n_1_old, fsp.u_dot_n_1, u_n_old)
        msh.transfer(u_dot_n_2_old, fsp.u_dot_n_2, u_n_old)

        # 4.4.3 transfer fluid fields
        msh.transfer(v_fl_n_old, fsp.v_fl_n, u_n_old)
        msh.transfer(v_fl_n_1_old, fsp.v_fl_n_1, u_n_old)
        msh.transfer(v_fl_n_2_old, fsp.v_fl_n_2, u_n_old)

        msh.transfer(phi_fl_old, fsp.phi_fl, u_n_old)

        msh.transfer(sigma_fl_n_12_old, fsp.sigma_fl_n_12, u_n_old)
        msh.transfer(sigma_fl_n_32_old, fsp.sigma_fl_n_32, u_n_old)

        #4.5 clean up

        del v_bar_old, w_bar_old, phi_old, v_n_old, w_n_old, U_n_12_old, nu_n_12_old, psi_n_12_old, mu_n_12_old, v_n_1_old, v_n_2_old, w_n_1_old, sigma_n_12_old, sigma_n_32_old, U_n_32_old, u_n_old, u_n_1_old, u_n_2_old, u_dot_n_old, u_dot_n_1_old, u_dot_n_2_old, v_fl_n_old, v_fl_n_1_old, v_fl_n_2_old, sigma_fl_n_12_old, sigma_fl_n_32_old
        gc.collect()

        print(f'{col.Fore.CYAN}... done.{col.Style.RESET_ALL}')


        #sign


    
    # 5. update  fields
    
    # 5.1 update the membrane problem 
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )

    fsp.v_n_2.assign( fsp.v_n_1 )
    fsp.v_n_1.assign( v_n_output )

    fsp.w_n_1.assign( w_n_output )

    fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )
    fsp.sigma_n_32.assign( fsp.sigma_n_12 )

    fsp.U_n_32.assign( U_n_12_output )


    # 2) update the mesh problem
    fsp.u_n_2.assign(fsp.u_n_1)
    fsp.u_n_1.assign(fsp.u_n)

    fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
    fsp.u_dot_n_1.assign(fsp.u_dot_n)
    
    # 3) update the fluid problem
    fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
    fsp.v_fl_n_1.assign(fsp.v_fl_n)

    fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)
    fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

    if step % rpam.parameters['print_out_stride'] == 0:
    # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        pr_sol.print_solution(t, step, dt)

    print(f'\t{(100.0 * (t / rpam.parameters["T"]))} %', flush=True)
    

print("... done.", flush=True)

fi.csvfile_bcs.close()
