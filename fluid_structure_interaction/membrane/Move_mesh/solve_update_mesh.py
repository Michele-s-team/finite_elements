
# """
# solve.py — complete file with remeshing integrated.
# """
# import dolfin
# from fenics import *
# import importlib
# import sys
# import numpy as np

# module_path = '/home/fenics/shared/modules'
# sys.path.append(module_path)

# import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
# import physics.fluid_mechanics as flu
# import function as fu
# import function_spaces as fsp
# import parameters.read.solution as rpam
# import physics.utils as phys
# import runtime_arguments as rarg
# import switch_problem as swi
# import variational_problem.utils as var_pr
# import print_out_solution as pr_sol
# import remesh as remesh_mod
# importlib.reload(remesh_mod) 


# dt = rpam.parameters['T'] / rpam.parameters['N']

# params = {'nonlinear_solver': 'newton',
#           'newton_solver': {
#               'linear_solver': 'superlu',
#               'absolute_tolerance': 1e-5,
#               'relative_tolerance': 1e-5,
#               'maximum_iterations': 1000,
#               'relaxation_parameter': 0.95,
#           }}

# PETScOptions.clear()
# PETScOptions.set('snes_type', 'newtonls')
# PETScOptions.set('snes_linesearch_type', 'bt')
# PETScOptions.set('snes_linesearch_maxstep', '1.0')
# PETScOptions.set('snes_atol', 1e-8)
# PETScOptions.set('snes_rtol', 1e-8)
# PETScOptions.set('snes_stol', 1e-8)
# PETScOptions.set('snes_max_it', 1000)
# PETScOptions.set('snes_monitor')
# PETScOptions.set('snes_max_funcs', 100000)

# rmsh = importlib.import_module(swi.rmsh)

# # -----------------------------------------------------------------------
# # Remesh parameters  <-- NEW BLOCK
# # -----------------------------------------------------------------------
# import tempfile
# path_to_meshfiles = tempfile.mkdtemp() + '/'
# L          = rmsh.parameters['L']
# h          = rmsh.parameters['h']
# gridsize   = 0.01
# bottom_tag = rmsh.parameters['sub_mesh_2_id']
# top_tag    = rmsh.parameters['sub_mesh_1_id']
# left_tag   = rmsh.parameters['line_sub_mesh_0_l_id']
# right_tag  = rmsh.parameters['line_sub_mesh_0_r_id']
# N_remesh   = 1

# # -----------------------------------------------------------------------
# # Initial setup
# # -----------------------------------------------------------------------
# fsp.var_tensor_sigma_fl.assign(project( flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1,rpam.parameters['eta_fluid']),fsp.Q_var_tensor_sigma_fl))
# fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl,fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])

# vp_membrane = importlib.import_module(swi.vp_membrane)

# v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split(deepcopy=True)
# fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
# fsp.U_dot_n_12.assign(project(
#     phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),
#     fsp.Q_U_dot_n_12))
# fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

# u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)
# fu.transfer_sub_mesh_to_mesh(u_fs_output,     fsp.u_fs_on_mesh)
# fu.transfer_sub_mesh_to_mesh(u_fs_dot_output, fsp.u_fs_dot_on_mesh)

# vp_mesh  = importlib.import_module(swi.vp_mesh)   # imported ONCE, never reloaded
# vp_fluid = importlib.import_module(swi.vp_fluid)
# pr_bc    = importlib.import_module(swi.prout_bc)

# dolfin.parameters["form_compiler"]["quadrature_degree"] = \
#     rpam.parameters['quadrature_degree']

# print("Input directory",  rarg.args.input_directory)
# print("Output directory", rarg.args.output_directory)

# # Initial conditions
# fsp.sigma_n_32.interpolate(
#     vp_membrane.sigma_n_32_0_Expression(element=fsp.Q_psi_n_12.ufl_element()))
# fsp.v_bar_0.interpolate(
#     vp_membrane.v_n_0_Expression(element=fsp.Q_v_bar.ufl_element()))
# fsp.v_n_0.interpolate(
#     vp_membrane.v_n_0_Expression(element=fsp.Q_v_n.ufl_element()))
# fsp.nu_n_12_0.interpolate(
#     vp_membrane.nu_n_12_0_Expression(element=fsp.Q_nu_n_12.ufl_element()))
# fsp.U_n_12_0.interpolate(
#     vp_membrane.U_n_12_0_Expression(element=fsp.Q_U_n_12.ufl_element()))
# fsp.sigma_fl_n_12.interpolate(
#     vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
# fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)
# fsp.assigner_mem.assign(fsp.psi_mem, [
#     fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0,
#     fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0])

# n_bottom = Constant((0.0, -1.0))
# print("v_bar_l =", rpam.parameters['v_bar_l'])
# print("dt =", dt)
# print("nu_n_12_0 min/max =", fsp.nu_n_12_0.vector().min(),
#                               fsp.nu_n_12_0.vector().max())
# print("mu_n_12_0 min/max =", fsp.mu_n_12_0.vector().min(),
#                               fsp.mu_n_12_0.vector().max())
# print("sigma_n_32 min/max =", fsp.sigma_n_32.vector().min(),fsp.sigma_n_32.vector().max())
  
# # -----------------------------------------------------------------------
# # Time loop
# # -----------------------------------------------------------------------
# t = 0
# step = 0

# for n in range(rpam.parameters['N']):
#     t += dt
#     step += 1

#     # ------------------------------------------------------------------
#     # Step 1: membrane
#     # ------------------------------------------------------------------
#     fsp.var_tensor_sigma_fl.assign(project(
#         flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1,
#                       rpam.parameters['eta_fluid']),
#         fsp.Q_var_tensor_sigma_fl))
#     fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl,
#                                   fsp.var_tensor_sigma_fl_on_mem,
#                                   rmsh.parameters['h'])
#     fsp.var_tensor_sigma_fl.vector()[:] = 0.0
#     fsp.var_tensor_sigma_fl_on_mem.vector()[:] = 0.0
#     vp_membrane = importlib.reload(vp_membrane)
#     var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem,
#                     vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)
#     print('Membrane done.', flush=True)
#     v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, \
#     U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = \
#     fsp.psi_mem.split(deepcopy=True)

#     print("AFTER MEM SOLVE step", step)
#     print("  w_bar min/max =", w_bar_output.vector().min(),
#                                 w_bar_output.vector().max())
#     print("  var_tensor_sigma_fl_on_mem min/max =",
#         fsp.var_tensor_sigma_fl_on_mem.vector().min(),
#         fsp.var_tensor_sigma_fl_on_mem.vector().max())
#     print("  sigma_n_32 min/max =", fsp.sigma_n_32.vector().min(),
#                                     fsp.sigma_n_32.vector().max())
#     print("  mu_n_12 min/max =", mu_n_12_output.vector().min(),
#                                 mu_n_12_output.vector().max())

#     # ------------------------------------------------------------------
#     # Step 2: mesh  (uses v_normal_proj updated at end of previous step)
#     # ------------------------------------------------------------------
#     v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, \
#         U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = \
#         fsp.psi_mem.split(deepcopy=True)
#     fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
#     fsp.U_dot_n_12.assign(project(
#         phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),
#         fsp.Q_U_dot_n_12))
#     fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

#     var_pr.solve_vp(vp_mesh.F_msh,     fsp.u_n,     vp_mesh.bcs_msh,
#                     fsp.J_u,     parameters=params)
#     var_pr.solve_vp(vp_mesh.F_msh_dot, fsp.u_dot_n, vp_mesh.bcs_msh_dot,
#                     fsp.J_u_dot, parameters=params)
#     print('Mesh done.', flush=True)

#     # ------------------------------------------------------------------
#     # Step 3: fluid
#     # ------------------------------------------------------------------
#     vp_fluid = importlib.reload(vp_fluid)
#     var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar,
#                     vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)
#     var_pr.solve_vp(vp_fluid.F_phi_fl,   fsp.phi_fl,
#                     vp_fluid.bc_phi_fl,   fsp.J_phi_fl,   parameters=params)
#     var_pr.solve_vp(vp_fluid.F_v_fl_n,   fsp.v_fl_n,
#                     [],                   fsp.J_v_fl_n,   parameters=params)
#     print('Fluid done.', flush=True)

#     # ------------------------------------------------------------------
#     # Step 4: update v_normal_proj in-place for next timestep's mesh BC
#     # ------------------------------------------------------------------
#     new_v_normal = project(dot(fsp.v_fl_n, n_bottom) * n_bottom, fsp.Q_u_dot)
#     vp_mesh.v_normal_proj.vector()[:] = new_v_normal.vector()[:]

#     # ------------------------------------------------------------------
#     # Step 5: history update (exactly once)
#     # ------------------------------------------------------------------
#     fsp.v_n_2.assign(fsp.v_n_1)
#     fsp.v_n_1.assign(v_n_output)
#     fsp.w_n_1.assign(w_n_output)
#     fsp.sigma_n_12.assign(fsp.sigma_n_32 - project(phi_output, fsp.Q_phi))
#     fsp.sigma_n_32.assign(fsp.sigma_n_12)
#     fsp.U_n_32.assign(U_n_12_output)

#     fsp.u_n.vector()[:] = fsp.u_n_1.vector()[:] + dt * fsp.u_dot_n.vector()[:]
#     fsp.u_n_2.assign(fsp.u_n_1)
#     fsp.u_n_1.assign(fsp.u_n)
#     fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
#     fsp.u_dot_n_1.assign(fsp.u_dot_n)

#     fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)
#     fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
#     fsp.v_fl_n_1.assign(fsp.v_fl_n)
#     fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

#     # ------------------------------------------------------------------
#     # Step 6: remesh every N_remesh steps  <-- NEW BLOCK
#     # ------------------------------------------------------------------
#     if step % N_remesh == 0:
#         print("--- Remeshing at step %i ---" % step, flush=True)

#         remesh_mod.do_remesh(
#             step=step,
#             path_to_meshfiles=path_to_meshfiles,
#             L=L, h=h, gridsize=gridsize,
#             bottom_tag=bottom_tag,
#             top_tag=top_tag,
#             left_tag=left_tag,
#             right_tag=right_tag,
#         )
#         # rebuild all mesh BCs and v_normal_proj on new spaces
#         remesh_mod.rebuild_mesh_bcs(vp_mesh, fsp, rmsh, n_bottom)

#         # vp_fluid also needs to be rebuilt since Q_v_fl_bar changed

#         print("--- Remesh done ---", flush=True)

#     # ------------------------------------------------------------------
#     # Diagnostics
#     # ------------------------------------------------------------------
#     pr_bc.print_bcs()
#     print("U min =", U_n_12_output.vector().min())
#     print("U max =", U_n_12_output.vector().max())
#     print("w min =", w_n_output.vector().min())
#     print("w max =", w_n_output.vector().max())
#     print("||u_dot_n|| =", norm(fsp.u_dot_n.vector(), 'l2'))
#     print("||u_n||     =", norm(fsp.u_n.vector(),     'l2'))

#     coords = rmsh.lmsh.sub_meshes[0].coordinates()
#     bot = np.where(np.abs(coords[:, 1] - coords[:, 1].min()) < 1e-12)[0]
#     print("bottom y min =", coords[bot, 1].min())
#     print("bottom y max =", coords[bot, 1].max())

#     if step % rpam.parameters['print_out_stride'] == 0:
#         pr_sol.print_solution(t, step, dt)

#     print(f'\t{100.0 * t / rpam.parameters["T"]:.1f} %', flush=True)

# # -----------------------------------------------------------------------
# # Post-loop diagnostics
# # -----------------------------------------------------------------------
# u_vec = U_n_12_output.vector().get_local()
# print("deformation amplitude =", u_vec.max() - u_vec.min())
# print("||u_dot_n|| =", norm(fsp.u_dot_n.vector(), 'l2'))
# print("||u_n||     =", norm(fsp.u_n.vector(),     'l2'))
# coords = rmsh.lmsh.sub_meshes[0].coordinates()
# print("mesh ymin =", coords[:, 1].min())
# print("mesh ymax =", coords[:, 1].max())




"""
This code solves for the dynamics of the Navier Stokes equations for a fluid
in a square whose top edge is a membrane. The coupled dynamics of membrane,
fluid and mesh are solved. The mesh is updated every timestep via harmonic
extension + ALE.move, replacing the elastic mesh BVP.

run with:
    rm -r solution; mkdir solution; python3 solve.py [problem] [mesh_path] [solution_path]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"
    SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/membrane/solution"
    rm -rf $SOLUTION_PATH
    python3 solve.py square_no_circle_line_a $MESH_PATH $SOLUTION_PATH
"""

import dolfin
from fenics import *
import importlib
import sys
import numpy as np

module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
import physics.fluid_mechanics as flu
import function as fu
import function_spaces as fsp
import parameters.read.solution as rpam
import physics.utils as phys
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr
import print_out_solution as pr_sol

dt = rpam.parameters['T'] / rpam.parameters['N']

params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-5,
                  'relative_tolerance': 1e-5,
                  'maximum_iterations': 1000,
                  'relaxation_parameter': 0.95,
              }
          }

PETScOptions.clear()
PETScOptions.set('snes_type', 'newtonls')
PETScOptions.set('snes_linesearch_type', 'bt')
PETScOptions.set('snes_linesearch_maxstep', '1.0')
PETScOptions.set('snes_atol', 1e-8)
PETScOptions.set('snes_rtol', 1e-8)
PETScOptions.set('snes_stol', 1e-8)
PETScOptions.set('snes_max_it', 1000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 100000)

rmsh = importlib.import_module(swi.rmsh)


# -----------------------------------------------------------------------
# Harmonic extension — replaces the elastic mesh BVP
# Solve nabla^2 u_h = 0 in the bulk with:
#   u_h = v_normal   on bottom (free surface)
#   u_h = 0          on top    (membrane fixed)
#   u_h.x = 0        on left/right walls
# Then ALE.move(mesh, dt * u_h) + mesh.smooth()
# -----------------------------------------------------------------------

def solve_harmonic_extension(v_normal_at_bottom):
    """Solve harmonic extension of bottom velocity into the bulk."""
    mesh = rmsh.lmsh.sub_meshes[0]
    mf   = rmsh.lmsh.mf_sub_meshes[0]
    V    = fsp.Q_u_dot

    u_h = TrialFunction(V)
    w   = TestFunction(V)

    a = inner(grad(u_h), grad(w)) * rmsh.dx_sub_mesh[0]
    L = inner(Constant((0.0, 0.0)), w) * rmsh.dx_sub_mesh[0]

    bc_top = DirichletBC(V, Constant((0.0, 0.0)),
                         mf, rmsh.parameters["sub_mesh_1_id"])
    bc_bot = DirichletBC(V, v_normal_at_bottom,
                         mf, rmsh.parameters["sub_mesh_2_id"])
    bc_l   = DirichletBC(V.sub(0), Constant(0.0),
                         mf, rmsh.parameters["line_sub_mesh_0_l_id"])
    bc_r   = DirichletBC(V.sub(0), Constant(0.0),
                         mf, rmsh.parameters["line_sub_mesh_0_r_id"])

    u_harmonic = Function(V)
    solve(a == L, u_harmonic, [bc_top, bc_bot, bc_l, bc_r])
    return u_harmonic


def move_mesh(u_harmonic):
    """Move mesh by dt*u_harmonic, smooth, update u_n and u_dot_n."""
    mesh = rmsh.lmsh.sub_meshes[0]

    disp = Function(fsp.Q_u_dot)
    disp.vector()[:] = dt * u_harmonic.vector()[:]

    ALE.move(mesh, disp)
    mesh.smooth()

    print("Mesh moved. ymin=%.6f  ymax=%.6f" % (
        mesh.coordinates()[:, 1].min(),
        mesh.coordinates()[:, 1].max()), flush=True)

    # update u_dot_n and u_n for ALE Jacobian in fluid forms
    fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
    fsp.u_dot_n_1.assign(fsp.u_dot_n)
    fsp.u_dot_n.assign(u_harmonic)

    fsp.u_n.vector()[:] = fsp.u_n_1.vector()[:] + dt * fsp.u_dot_n.vector()[:]
    fsp.u_n_2.assign(fsp.u_n_1)
    fsp.u_n_1.assign(fsp.u_n)


# -----------------------------------------------------------------------
# Initial setup (before the loop)
# -----------------------------------------------------------------------

# 1) membrane — compute initial fluid stress (zero at t=0)
fsp.var_tensor_sigma_fl.assign(project(
    flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1,
                  rpam.parameters['eta_fluid']),
    fsp.Q_var_tensor_sigma_fl))
fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl,
                              fsp.var_tensor_sigma_fl_on_mem,
                              rmsh.parameters['h'])

# zero fluid stress at t=0 — fluid not yet solved
fsp.var_tensor_sigma_fl.vector()[:] = 0.0
fsp.var_tensor_sigma_fl_on_mem.vector()[:] = 0.0

vp_membrane = importlib.import_module(swi.vp_membrane)

# 2) membrane displacement onto mesh (for completeness at t=0)
v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, \
    U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = \
    fsp.psi_mem.split(deepcopy=True)
fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
fsp.U_dot_n_12.assign(project(
    phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),
    fsp.Q_U_dot_n_12))
fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

# 3) free surface initial projection
u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)
fu.transfer_sub_mesh_to_mesh(u_fs_output,     fsp.u_fs_on_mesh)
fu.transfer_sub_mesh_to_mesh(u_fs_dot_output, fsp.u_fs_dot_on_mesh)

# 4) fluid
vp_fluid = importlib.import_module(swi.vp_fluid)

pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = \
    rpam.parameters['quadrature_degree']

print("Input directory",  rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

# -----------------------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------------------
fsp.sigma_n_32.interpolate(
    vp_membrane.sigma_n_32_0_Expression(element=fsp.Q_psi_n_12.ufl_element()))

fsp.v_bar_0.interpolate(
    vp_membrane.v_n_0_Expression(element=fsp.Q_v_bar.ufl_element()))
fsp.v_n_0.interpolate(
    vp_membrane.v_n_0_Expression(element=fsp.Q_v_n.ufl_element()))
fsp.nu_n_12_0.interpolate(
    vp_membrane.nu_n_12_0_Expression(element=fsp.Q_nu_n_12.ufl_element()))
fsp.U_n_12_0.interpolate(
    vp_membrane.U_n_12_0_Expression(element=fsp.Q_U_n_12.ufl_element()))

fsp.sigma_fl_n_12.interpolate(
    vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

fsp.assigner_mem.assign(fsp.psi_mem, [
    fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0,
    fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0])

# normal vector for bottom boundary
n_bottom = Constant((0.0, -1.0))

# -----------------------------------------------------------------------
# Time loop
# -----------------------------------------------------------------------
t    = 0
step = 0

for n in range(rpam.parameters['N']):
    t    += dt
    step += 1

    # ------------------------------------------------------------------
    # Step 1: membrane
    # ------------------------------------------------------------------
    fsp.var_tensor_sigma_fl.assign(project(
        flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1,
                      rpam.parameters['eta_fluid']),
        fsp.Q_var_tensor_sigma_fl))
    fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl,
                                  fsp.var_tensor_sigma_fl_on_mem,
                                  rmsh.parameters['h'])

    vp_membrane = importlib.reload(vp_membrane)
    var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem,
                    vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)
    print('Membrane done.', flush=True)

    # ------------------------------------------------------------------
    # Step 2: fluid (three-step pressure-correction splitting)
    # ------------------------------------------------------------------
    vp_fluid = importlib.reload(vp_fluid)
    var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar,
                    vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)
    var_pr.solve_vp(vp_fluid.F_phi_fl,   fsp.phi_fl,
                    vp_fluid.bc_phi_fl,   fsp.J_phi_fl,   parameters=params)
    var_pr.solve_vp(vp_fluid.F_v_fl_n,   fsp.v_fl_n,
                    [],                   fsp.J_v_fl_n,   parameters=params)
    print('Fluid done.', flush=True)

    # ------------------------------------------------------------------
    # Step 3: harmonic extension + ALE mesh move (every timestep)
    # ------------------------------------------------------------------
    v_normal_bottom = project(
        dot(fsp.v_fl_n, n_bottom) * n_bottom, fsp.Q_u_dot)
    u_harmonic = solve_harmonic_extension(v_normal_bottom)
    move_mesh(u_harmonic)
    print('Mesh done.', flush=True)

    # ------------------------------------------------------------------
    # Step 4: print BCs
    # ------------------------------------------------------------------
    pr_bc.print_bcs()

    # ------------------------------------------------------------------
    # Step 5: update history fields — exactly once per timestep
    # ------------------------------------------------------------------

    # split membrane solution
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, \
        U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = \
        fsp.psi_mem.split(deepcopy=True)

    # transfer membrane displacement and velocity to bulk mesh
    fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
    fsp.U_dot_n_12.assign(project(
        phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),
        fsp.Q_U_dot_n_12))
    fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

    print("U min =", U_n_12_output.vector().min())
    print("U max =", U_n_12_output.vector().max())
    print("w min =", w_n_output.vector().min())
    print("w max =", w_n_output.vector().max())

    # membrane history
    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(v_n_output)
    fsp.w_n_1.assign(w_n_output)
    fsp.sigma_n_12.assign(fsp.sigma_n_32 - project(phi_output, fsp.Q_phi))
    fsp.sigma_n_32.assign(fsp.sigma_n_12)
    fsp.U_n_32.assign(U_n_12_output)

    # mesh history updated inside move_mesh()

    # fluid history
    fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)
    fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
    fsp.v_fl_n_1.assign(fsp.v_fl_n)
    fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

    # ------------------------------------------------------------------
    # Print output and progress
    # ------------------------------------------------------------------
    if step % rpam.parameters['print_out_stride'] == 0:
        pr_sol.print_solution(t, step, dt)

    print(f'\t{(100.0 * (t / rpam.parameters["T"]))} %', flush=True)

    coords = rmsh.lmsh.sub_meshes[0].coordinates()
    bottom = np.where(np.abs(coords[:, 1] - coords[:, 1].min()) < 1e-12)[0]
    print("bottom y min =", coords[bottom, 1].min())
    print("bottom y max =", coords[bottom, 1].max())

# -----------------------------------------------------------------------
# Post-loop diagnostics
# -----------------------------------------------------------------------
u_vec = U_n_12_output.vector().get_local()
print("deformation amplitude =", u_vec.max() - u_vec.min())

print("||U_dot_n_12_on_mesh|| =",
      norm(fsp.U_dot_n_12_on_mesh.vector(), 'l2'))
print("||u_dot_n|| =",
      norm(fsp.u_dot_n.vector(), 'l2'))
print("||u_n|| =",
      norm(fsp.u_n.vector(), 'l2'))

print("... done.", flush=True)

u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)
print("||u_fs_output|| =",     norm(u_fs_output))
print("||u_fs_dot_output|| =", norm(u_fs_dot_output))
print("max free surface displacement =",
      np.max(np.abs(u_fs_output.vector().get_local())))

print("psi min =", psi_n_12_output.vector().min())
print("psi max =", psi_n_12_output.vector().max())
print("mu min =",  mu_n_12_output.vector().min())
print("mu max =",  mu_n_12_output.vector().max())

coords = rmsh.lmsh.sub_meshes[0].coordinates()
print("mesh ymin =", coords[:, 1].min())
print("mesh ymax =", coords[:, 1].max())