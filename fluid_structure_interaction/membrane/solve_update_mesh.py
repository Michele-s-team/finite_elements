
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
    pr_bc.print_bcs()
    pr_bc.print_bcs()
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
"""
solve.py — complete file with gmsh remesh every timestep.

Pattern per timestep (following stokes_FEM.py):
  1. Solve membrane
  2. Solve fluid on current mesh
  3. Harmonic extension: nabla^2 u_h = 0, BC = v_fl_n.n at bottom
  4. ALE.move(mesh, dt * u_h) + mesh.smooth()  — move mesh for next step
  5. gmsh remesh from moved boundary coords     — rebuild clean mesh
  6. Reload mesh, rebuild function spaces, interpolate fields
  7. Update history

run with:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"
    SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/membrane/solution"
    rm -rf $SOLUTION_PATH
    python3 solve.py square_no_circle_line_a $MESH_PATH $SOLUTION_PATH
"""

import dolfin
from fenics import *
import importlib
import sys
import os
import glob
import numpy as np
import gmsh
import tempfile
from dolfin import LagrangeInterpolator

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

dt   = rpam.parameters['T'] / rpam.parameters['N']
rmsh = importlib.import_module(swi.rmsh)

params = {'nonlinear_solver': 'newton',
          'newton_solver': {
              'linear_solver': 'superlu',
              'absolute_tolerance': 1e-5,
              'relative_tolerance': 1e-5,
              'maximum_iterations': 1000,
              'relaxation_parameter': 0.95,
          }}

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

# temp dir for gmsh files
path_to_meshfiles = tempfile.mkdtemp() + '/'

# mesh geometry parameters
L          = rmsh.parameters['L']
h          = rmsh.parameters['h']
gridsize   = 0.05
bottom_tag = rmsh.parameters['sub_mesh_2_id']
top_tag    = rmsh.parameters['sub_mesh_1_id']
left_tag   = rmsh.parameters['line_sub_mesh_0_l_id']
right_tag  = rmsh.parameters['line_sub_mesh_0_r_id']

n_bottom = Constant((0.0, -1.0))


# =======================================================================
# Helper functions
# =======================================================================

def extract_bottom_coords():
    """Extract x-sorted bottom boundary node coordinates from current mesh."""
    mesh = rmsh.lmsh.sub_meshes[0]
    mf   = rmsh.lmsh.mf_sub_meshes[0]
    V    = FunctionSpace(mesh, "CG", 1)
    v2d  = vertex_to_dof_map(V)
    dofs = []
    for facet in facets(mesh):
        if mf[facet.index()] == bottom_tag:
            for vertex in facet.entities(0):
                dofs.append(v2d[vertex])
    unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
    coords = V.tabulate_dof_coordinates()[unique_dofs]
    return coords[np.argsort(coords[:, 0])]


def build_gmsh_mesh(bottom_coords, step):
    """
    Build new gmsh mesh from displaced bottom boundary coords.
    Bottom = spline through bottom_coords (already moved by ALE).
    Top    = flat line at y = h.
    Returns xml_base path (without .xml extension).
    """
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("domain_%i" % step)

    bottom_pts = []
    for (x, y) in bottom_coords:
        bottom_pts.append(gmsh.model.geo.addPoint(x, y, 0, gridsize))

    p_tl = gmsh.model.geo.addPoint(0.0, h, 0, gridsize)
    p_tr = gmsh.model.geo.addPoint(L,   h, 0, gridsize)

    line_bottom = gmsh.model.geo.addSpline(bottom_pts)
    line_right  = gmsh.model.geo.addLine(bottom_pts[-1], p_tr)
    line_top    = gmsh.model.geo.addLine(p_tr, p_tl)
    line_left   = gmsh.model.geo.addLine(p_tl, bottom_pts[0])

    cl   = gmsh.model.geo.addCurveLoop([line_bottom, line_right, line_top, line_left])
    surf = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.addPhysicalGroup(2, [surf],        tag=1)
    gmsh.model.geo.addPhysicalGroup(1, [line_bottom], tag=bottom_tag)
    gmsh.model.geo.addPhysicalGroup(1, [line_top],    tag=top_tag)
    gmsh.model.geo.addPhysicalGroup(1, [line_left],   tag=left_tag)
    gmsh.model.geo.addPhysicalGroup(1, [line_right],  tag=right_tag)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [line_bottom])
    gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", 1000)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField",  1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin",  gridsize / 5)
    gmsh.model.mesh.field.setNumber(2, "SizeMax",  gridsize)
    gmsh.model.mesh.field.setNumber(2, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax",  h / 2)
    gmsh.model.mesh.field.add("Min", 7)
    gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(7)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)

    gmsh.model.mesh.generate(dim=2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    msh_file = path_to_meshfiles + "mesh_%i.msh" % step
    gmsh.write(msh_file)
    gmsh.finalize()

    xml_base = path_to_meshfiles + "domain"
    os.system("dolfin-convert %s %s.xml" % (msh_file, xml_base))
    return xml_base


def load_new_mesh(xml_base):
    """Load new mesh and facet MeshFunction, update rmsh."""
    new_mesh = Mesh(xml_base + ".xml")
    rmsh.lmsh.sub_meshes[0] = new_mesh

    candidates = (glob.glob(xml_base + "_facet_region.xml") +
                  glob.glob(xml_base + "_physical_region.xml"))
    if not candidates:
        raise RuntimeError("No facet XML found in %s" % path_to_meshfiles)
    new_mf = MeshFunction("size_t", new_mesh, candidates[0])
    rmsh.lmsh.mf_sub_meshes[0] = new_mf

    # rebuild integration measures
    new_dx = Measure("dx", domain=new_mesh)
    new_ds = Measure("ds", domain=new_mesh, subdomain_data=new_mf)
    rmsh.dx_sub_mesh[0] = new_dx
    rmsh.ds_sub_mesh[0] = {
        'ds':    new_ds,
        'ds_t':  new_ds(top_tag),
        'ds_b':  new_ds(bottom_tag),
        'ds_l':  new_ds(left_tag),
        'ds_r':  new_ds(right_tag),
        'ds_tb': new_ds(top_tag)  + new_ds(bottom_tag),
        'ds_lr': new_ds(left_tag) + new_ds(right_tag),
    }
    return new_mesh, new_mf


def rebuild_fluid_spaces(new_mesh):
    """Rebuild fluid/mesh function spaces and functions on new mesh."""
    fsp.Q_u             = VectorFunctionSpace(new_mesh, 'P', 1)
    fsp.Q_u_dot         = VectorFunctionSpace(new_mesh, 'P', 1)
    fsp.Q_v_fl          = VectorFunctionSpace(new_mesh, 'P', 2)
    fsp.Q_v_fl_bar      = VectorFunctionSpace(new_mesh, 'P', 2)
    fsp.Q_phi_fl        = FunctionSpace(new_mesh, 'P', 1)
    tdeg = fsp.Q_var_tensor_sigma_fl.ufl_element().degree()
    fsp.Q_var_tensor_sigma_fl = TensorFunctionSpace(new_mesh, 'P', tdeg, shape=(2, 2))

    fsp.u_n             = Function(fsp.Q_u)
    fsp.u_n_1           = Function(fsp.Q_u)
    fsp.u_n_2           = Function(fsp.Q_u)
    fsp.u_dot_n         = Function(fsp.Q_u_dot)
    fsp.u_dot_n_1       = Function(fsp.Q_u_dot)
    fsp.u_dot_n_2       = Function(fsp.Q_u_dot)
    fsp.v_fl_n          = Function(fsp.Q_v_fl)
    fsp.v_fl_n_1        = Function(fsp.Q_v_fl)
    fsp.v_fl_n_2        = Function(fsp.Q_v_fl)
    fsp.v_fl_bar        = Function(fsp.Q_v_fl_bar)
    fsp.sigma_fl_n_12   = Function(fsp.Q_phi_fl)
    fsp.sigma_fl_n_32   = Function(fsp.Q_phi_fl)
    fsp.phi_fl          = Function(fsp.Q_phi_fl)
    fsp.var_tensor_sigma_fl = Function(fsp.Q_var_tensor_sigma_fl)

    fsp.J_u             = TrialFunction(fsp.Q_u)
    fsp.J_u_dot         = TrialFunction(fsp.Q_u_dot)
    fsp.nu_u            = TestFunction(fsp.Q_u)
    fsp.nu_u_dot        = TestFunction(fsp.Q_u_dot)
    fsp.J_v_fl_bar      = TrialFunction(fsp.Q_v_fl_bar)
    fsp.J_v_fl_n        = TrialFunction(fsp.Q_v_fl)
    fsp.J_phi_fl        = TrialFunction(fsp.Q_phi_fl)
    fsp.nu_v_fl_n       = TestFunction(fsp.Q_v_fl)
    fsp.nu_v_fl_bar     = TestFunction(fsp.Q_v_fl_bar)
    fsp.nu_phi_fl       = TestFunction(fsp.Q_phi_fl)
    fsp.V_fl            = (fsp.v_fl_n_1 + fsp.v_fl_bar) / 2.0


def interp(old_func, new_space):
    """Interpolate old_func onto new_space (non-matching meshes)."""
    f = Function(new_space)
    LagrangeInterpolator.interpolate(f, old_func)
    return f


def interpolate_fields_to_new_mesh(old):
    """Interpolate saved fluid fields onto new function spaces."""
    parameters['allow_extrapolation'] = True
    fsp.v_fl_n_1.assign(     interp(old['v_fl_n_1'],      fsp.Q_v_fl))
    fsp.v_fl_n_2.assign(     interp(old['v_fl_n_2'],      fsp.Q_v_fl))
    fsp.v_fl_bar.assign(     interp(old['v_fl_bar'],      fsp.Q_v_fl_bar))
    fsp.sigma_fl_n_12.assign(interp(old['sigma_fl_n_12'], fsp.Q_phi_fl))
    fsp.sigma_fl_n_32.assign(interp(old['sigma_fl_n_32'], fsp.Q_phi_fl))
    fsp.phi_fl.assign(       interp(old['phi_fl'],        fsp.Q_phi_fl))
    parameters['allow_extrapolation'] = False
    # mesh displacement resets to zero — new mesh IS the new reference config
    fsp.u_n.vector()[:]       = 0.0
    fsp.u_n_1.vector()[:]     = 0.0
    fsp.u_n_2.vector()[:]     = 0.0
    fsp.u_dot_n.vector()[:]   = 0.0
    fsp.u_dot_n_1.vector()[:] = 0.0
    fsp.u_dot_n_2.vector()[:] = 0.0


def solve_harmonic_extension(v_normal_at_bottom):
    """Solve nabla^2 u_h = 0 with v_normal_at_bottom as bottom BC."""
    mf = rmsh.lmsh.mf_sub_meshes[0]
    V  = fsp.Q_u_dot
    u_h = TrialFunction(V)
    w   = TestFunction(V)
    a = inner(grad(u_h), grad(w)) * rmsh.dx_sub_mesh[0]
    L = inner(Constant((0.0, 0.0)), w) * rmsh.dx_sub_mesh[0]
    bcs = [
        DirichletBC(V, Constant((0.0, 0.0)), mf, top_tag),
        DirichletBC(V, v_normal_at_bottom,   mf, bottom_tag),
        DirichletBC(V.sub(0), Constant(0.0), mf, left_tag),
        DirichletBC(V.sub(0), Constant(0.0), mf, right_tag),
    ]
    u_harmonic = Function(V)
    solve(a == L, u_harmonic, bcs)
    return u_harmonic


def rebuild_fluid_bcs(vp_fluid, vp_mesh=None):
    """After remesh, reload fluid module to rebuild its BCs on new spaces."""
    return importlib.reload(vp_fluid)


# =======================================================================
# Initial setup
# =======================================================================

# zero fluid stress at t=0
fsp.var_tensor_sigma_fl.vector()[:] = 0.0
fsp.var_tensor_sigma_fl_on_mem.vector()[:] = 0.0

vp_membrane = importlib.import_module(swi.vp_membrane)

v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, \
    U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = \
    fsp.psi_mem.split(deepcopy=True)
fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
fsp.U_dot_n_12.assign(project(
    phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),
    fsp.Q_U_dot_n_12))
fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)
fu.transfer_sub_mesh_to_mesh(u_fs_output,     fsp.u_fs_on_mesh)
fu.transfer_sub_mesh_to_mesh(u_fs_dot_output, fsp.u_fs_dot_on_mesh)

vp_fluid = importlib.import_module(swi.vp_fluid)
pr_bc    = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = \
    rpam.parameters['quadrature_degree']

print("Input directory",  rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

# =======================================================================
# Initial conditions
# =======================================================================
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

# =======================================================================
# Time loop
# =======================================================================
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
    var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar,
                    vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)
    var_pr.solve_vp(vp_fluid.F_phi_fl,   fsp.phi_fl,
                    vp_fluid.bc_phi_fl,   fsp.J_phi_fl,   parameters=params)
    var_pr.solve_vp(vp_fluid.F_v_fl_n,   fsp.v_fl_n,
                    [],                   fsp.J_v_fl_n,   parameters=params)
    print('Fluid done.', flush=True)
    pr_bc.print_bcs()


    # ------------------------------------------------------------------
    # Step 3: harmonic extension
    # Solve nabla^2 u_h = 0 with bottom BC = v_fl_n . n
    # ------------------------------------------------------------------
    v_normal_bottom = project(
        dot(fsp.v_fl_n, n_bottom) * n_bottom, fsp.Q_u_dot)
    u_harmonic = solve_harmonic_extension(v_normal_bottom)
    print('Harmonic extension done.', flush=True)

    # ------------------------------------------------------------------
    # Step 4: ALE.move + mesh.smooth
    # Physically move mesh nodes by dt * u_harmonic
    # ------------------------------------------------------------------
    disp = Function(fsp.Q_u_dot)
    disp.vector()[:] = dt * u_harmonic.vector()[:]
    ALE.move(rmsh.lmsh.sub_meshes[0], disp)
    rmsh.lmsh.sub_meshes[0].smooth()
    print("ALE.move done. ymin=%.6f  ymax=%.6f" % (
        rmsh.lmsh.sub_meshes[0].coordinates()[:, 1].min(),
        rmsh.lmsh.sub_meshes[0].coordinates()[:, 1].max()), flush=True)

    # ------------------------------------------------------------------
    # Step 5: gmsh remesh from moved boundary coords
    # ------------------------------------------------------------------
    # save fluid fields before rebuilding spaces
    old = {
        'v_fl_n_1':      fsp.v_fl_n_1.copy(deepcopy=True),
        'v_fl_n_2':      fsp.v_fl_n_2.copy(deepcopy=True),
        'v_fl_bar':      fsp.v_fl_bar.copy(deepcopy=True),
        'sigma_fl_n_12': fsp.sigma_fl_n_12.copy(deepcopy=True),
        'sigma_fl_n_32': fsp.sigma_fl_n_32.copy(deepcopy=True),
        'phi_fl':        fsp.phi_fl.copy(deepcopy=True),
    }

    # extract bottom coords from moved mesh, build new gmsh mesh
    bottom_coords = extract_bottom_coords()
    xml_base      = build_gmsh_mesh(bottom_coords, step)
    new_mesh, new_mf = load_new_mesh(xml_base)

    # rebuild function spaces and interpolate fields
    rebuild_fluid_spaces(new_mesh)
    interpolate_fields_to_new_mesh(old)

    # reload fluid module so its BCs and forms use new function spaces

    print("Remesh done. New mesh: %i cells." % new_mesh.num_cells(), flush=True)

    # ------------------------------------------------------------------
    # Step 6: history update — exactly once per timestep
    # ------------------------------------------------------------------
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, \
        U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = \
        fsp.psi_mem.split(deepcopy=True)

    fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
    fsp.U_dot_n_12.assign(project(
        phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),
        fsp.Q_U_dot_n_12))
    fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

    # membrane history
    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(v_n_output)
    fsp.w_n_1.assign(w_n_output)
    fsp.sigma_n_12.assign(fsp.sigma_n_32 - project(phi_output, fsp.Q_phi))
    fsp.sigma_n_32.assign(fsp.sigma_n_12)
    fsp.U_n_32.assign(U_n_12_output)

    # fluid history (u_n reset to 0 inside interpolate_fields_to_new_mesh)
    fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)
    fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
    fsp.v_fl_n_1.assign(fsp.v_fl_n)
    fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    print("U   min = %.4e  max = %.4e" % (
        U_n_12_output.vector().min(), U_n_12_output.vector().max()))
    print("w   min = %.4e  max = %.4e" % (
        w_n_output.vector().min(), w_n_output.vector().max()))
    print("psi min = %.4e  max = %.4e" % (
        psi_n_12_output.vector().min(), psi_n_12_output.vector().max()))
    print("mu  min = %.4e  max = %.4e" % (
        mu_n_12_output.vector().min(), mu_n_12_output.vector().max()))
    print("sigma_n_32 min = %.4e  max = %.4e" % (
        fsp.sigma_n_32.vector().min(), fsp.sigma_n_32.vector().max()))
    print("v_fl_n  L2 = %.4e" % norm(fsp.v_fl_n.vector(),  'l2'))
    print("u_dot_n L2 = %.4e" % norm(fsp.u_dot_n.vector(), 'l2'))
    print("u_n     L2 = %.4e" % norm(fsp.u_n.vector(),     'l2'))

    coords = rmsh.lmsh.sub_meshes[0].coordinates()
    bot    = np.where(np.abs(coords[:, 1] - coords[:, 1].min()) < 1e-12)[0]
    print("bottom y: min = %.6f  max = %.6f" % (
        coords[bot, 1].min(), coords[bot, 1].max()))

    if step % rpam.parameters['print_out_stride'] == 0:
        pr_sol.print_solution(t, step, dt)

    print(f'\t{100.0 * t / rpam.parameters["T"]:.1f} %', flush=True)

# =======================================================================
# Post-loop diagnostics
# =======================================================================
u_vec = U_n_12_output.vector().get_local()
print("deformation amplitude =", u_vec.max() - u_vec.min())
print("||U_dot_n_12_on_mesh|| =", norm(fsp.U_dot_n_12_on_mesh.vector(), 'l2'))
print("||u_dot_n|| =", norm(fsp.u_dot_n.vector(), 'l2'))
print("||u_n||     =", norm(fsp.u_n.vector(),     'l2'))
print("... done.", flush=True)

u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)
print("||u_fs_output||     =", norm(u_fs_output))
print("||u_fs_dot_output|| =", norm(u_fs_dot_output))
print("max free surface displacement =",
      np.max(np.abs(u_fs_output.vector().get_local())))
print("psi min =", psi_n_12_output.vector().min())
print("psi max =", psi_n_12_output.vector().max())
print("mu  min =", mu_n_12_output.vector().min())
print("mu  max =", mu_n_12_output.vector().max())

coords = rmsh.lmsh.sub_meshes[0].coordinates()
print("mesh ymin =", coords[:, 1].min())
print("mesh ymax =", coords[:, 1].max())