"""
solve.py — fluid-structure interaction with Helfrich membrane and free surface.
Stress output follows the approach :
  - project full stress tensor onto CG1 on bulk mesh
  - sample at membrane node coordinates via point evaluation
  - compute sig_nn, sig_nt, traction via numpy
  - save one CSV per timestep + XDMF for ParaView

run with:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"
    SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/membrane/solution"
    rm -rf $SOLUTION_PATH python3 solve.py square_no_circle_line_a $MESH_PATH $SOLUTION_PATH
"""

import dolfin
from fenics import *
import importlib
import sys
import numpy as np
import os

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

rmsh = importlib.import_module(swi.rmsh)

# ── initial projections ───────────────────────────────────────────────────────
fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1,rpam.parameters['eta_fluid']),fsp.Q_var_tensor_sigma_fl))
fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl,fsp.var_tensor_sigma_fl_on_mem,rmsh.parameters['h'])

vp_membrane = importlib.import_module(swi.vp_membrane)

v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split(deepcopy=True)
fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),fsp.Q_U_dot_n_12))
fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)
fu.transfer_sub_mesh_to_mesh(u_fs_output,     fsp.u_fs_on_mesh)
fu.transfer_sub_mesh_to_mesh(u_fs_dot_output, fsp.u_fs_dot_on_mesh)

vp_mesh  = importlib.import_module(swi.vp_mesh)
vp_fluid = importlib.import_module(swi.vp_fluid)
pr_bc    = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = \
    rpam.parameters['quadrature_degree']

print("Input directory",  rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

fsp.sigma_n_32.interpolate(vp_membrane.sigma_n_32_0_Expression(element=fsp.Q_psi_n_12.ufl_element()))
fsp.v_bar_0.interpolate(vp_membrane.v_n_0_Expression(element=fsp.Q_v_bar.ufl_element()))
fsp.v_n_0.interpolate(vp_membrane.v_n_0_Expression(element=fsp.Q_v_n.ufl_element()))
fsp.nu_n_12_0.interpolate(vp_membrane.nu_n_12_0_Expression(element=fsp.Q_nu_n_12.ufl_element()))
fsp.U_n_12_0.interpolate(vp_membrane.U_n_12_0_Expression(element=fsp.Q_U_n_12.ufl_element()))
fsp.sigma_fl_n_12.interpolate(vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

fsp.assigner_mem.assign(fsp.psi_mem, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0,fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0])

n_bottom = Constant((0.0, -1.0))

# Function spaces on bulk mesh for stress sampling
W_CG1 = TensorFunctionSpace(rmsh.lmsh.sub_meshes[0], 'CG', 1)
V_CG1 = VectorFunctionSpace(rmsh.lmsh.sub_meshes[0], 'CG', 1)
K_CG1 = FunctionSpace(rmsh.lmsh.sub_meshes[0], 'CG', 1)

stress_bulk  = Function(W_CG1, name='Total_Stress')
trac_bulk    = Function(V_CG1, name='Traction')
sig_nn_bulk  = Function(K_CG1, name='Sig_nn')
sig_nt_bulk  = Function(K_CG1, name='Sig_nt')

# single XDMF file for all stress fields (like stokes_FEM.py xdmf_stokes)
xdmf_stress = XDMFFile(rarg.args.output_directory + '/stress.xdmf')
xdmf_stress.parameters['flush_output'] = True
xdmf_stress.parameters['functions_share_mesh'] = True

# get sorted membrane node coordinates for point sampling
# sub_mesh_1 is a 1D interval mesh: coordinates are shape (N,1), just x values
# the membrane lives at y = h in the bulk mesh
mem_mesh = rmsh.lmsh.sub_meshes[1]
mem_x = mem_mesh.coordinates().flatten()   # shape (N,) just x
sort_idx = np.argsort(mem_x)
mem_x_sorted = mem_x[sort_idx]                     

# CSV output directory
csv_dir = rarg.args.output_directory + '/surface_stresses/'
os.makedirs(csv_dir, exist_ok=True)

t    = 0
step = 0

for n in range(rpam.parameters['N']):
    t    += dt
    step += 1

    # ------------------------------------------------------------------
    # Step 1: membrane
    # ------------------------------------------------------------------
    fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1,rpam.parameters['eta_fluid']),fsp.Q_var_tensor_sigma_fl))
    fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl,fsp.var_tensor_sigma_fl_on_mem,rmsh.parameters['h'])

    vp_membrane = importlib.reload(vp_membrane)
    var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem,
                    vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)
    print('Membrane done.', flush=True)

    # ------------------------------------------------------------------
    # Step 2: mesh
    # ------------------------------------------------------------------
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split(deepcopy=True)
    fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
    fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),fsp.Q_U_dot_n_12))
    fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

    vp_mesh = importlib.reload(vp_mesh)
    var_pr.solve_vp(vp_mesh.F_msh,fsp.u_n,vp_mesh.bcs_msh,fsp.J_u,parameters=params)
    var_pr.solve_vp(vp_mesh.F_msh_dot, fsp.u_dot_n, vp_mesh.bcs_msh_dot, fsp.J_u_dot, parameters=params)

    new_v_normal = project(dot(fsp.v_fl_n, n_bottom)*n_bottom, fsp.Q_u_dot)
    vp_mesh.v_normal_proj.vector()[:] = new_v_normal.vector()[:]
    fsp.u_fs_on_mesh.vector()[:] = fsp.u_n_1.vector()[:] + dt * new_v_normal.vector()[:]
    print('Mesh done.', flush=True)

    # ------------------------------------------------------------------
    # Step 3: fluid
    # ------------------------------------------------------------------
    vp_fluid = importlib.reload(vp_fluid)
    var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar,vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)
    var_pr.solve_vp(vp_fluid.F_phi_fl, fsp.phi_fl,vp_fluid.bc_phi_fl, fsp.J_phi_fl,   parameters=params)
    var_pr.solve_vp(vp_fluid.F_v_fl_n, fsp.v_fl_n,[], fsp.J_v_fl_n, parameters=params)
    print('Fluid done.', flush=True)

    # ------------------------------------------------------------------
    # Step 4: stress and geometry on membrane:
    #   1. project full stress tensor onto CG1 on bulk mesh
    #   2. sample at membrane node coordinates via point evaluation
    #   3. compute sig_nn, sig_nt from sampled values using numpy
    #   4. save CSV per step + XDMF for ParaView
    # ------------------------------------------------------------------

    # 4a. recompute stress with fresh v_fl_n
    sigma_r_F = flu.sigma_ale(fsp.v_fl_n, fsp.sigma_fl_n_32, fsp.u_n_1,rpam.parameters['eta_fluid'])

    # 4b. project full stress tensor onto CG1 on bulk mesh
    stress_bulk.assign(project(sigma_r_F, W_CG1))

    # 4c. traction vector: f = sigma . n_top  where n_top = (0,1)
    n_top = Constant((0.0, 1.0))   # outward normal of fluid at top membrane
    traction_vec = dot(sigma_r_F, n_top)
    trac_bulk.assign(project(traction_vec, V_CG1))

    # 4d. extract stress components as scalar CG1 fields for sampling
    sig_xx_fn = project(stress_bulk[0, 0], K_CG1)
    sig_xy_fn = project(stress_bulk[0, 1], K_CG1)
    sig_yy_fn = project(stress_bulk[1, 1], K_CG1)

    # 4e. sample at sorted membrane node coordinates (point evaluation)
    # also sample the current membrane normal and tangent from U_n_12_output
    sig_xx_v = []
    sig_xy_v = []
    sig_yy_v = []
    n_x_v    = []
    n_y_v    = []
    t_x_v    = []
    t_y_v    = []

    h_mem = float(rmsh.parameters['h'])
    L_mem = float(rmsh.parameters['L'])
    A_mem = float(rmsh.parameters['A'])
    n_mem_p = float(rmsh.parameters['n'])
    lmda  = float(rmsh.parameters['lmda'])
    q_mem = 4 * np.pi / lmda    # wavenumber of cosine shape
    eps   = 1e-7

    # allow extrapolation so boundary points do not fail
    sig_xx_fn.set_allow_extrapolation(True)
    sig_xy_fn.set_allow_extrapolation(True)
    sig_yy_fn.set_allow_extrapolation(True)

    for x in mem_x_sorted:
        x = float(x)
        # membrane current y position includes cosine shape + displacement
        y_mem = h_mem + A_mem * np.cos(q_mem * x)
        # sample stress on bulk mesh at membrane position
        xpt = (x, y_mem)

        sig_xx_v.append(sig_xx_fn(xpt))
        sig_xy_v.append(sig_xy_fn(xpt))
        sig_yy_v.append(sig_yy_fn(xpt))

        # normal and tangent from FULL membrane geometry:
        # X^c = (x + U[0], h + A*cos(q*x) + U[1])
        # dX^c/dx = (1 + dU0/dx,  -A*q*sin(q*x) + dU1/dx)
        x1d_p = (min(x + eps, L_mem),)
        x1d_m = (max(x - eps, 0.0),)
        dU0dx = (U_n_12_output[0](x1d_p) - U_n_12_output[0](x1d_m)) / (2*eps)
        dU1dx = (U_n_12_output[1](x1d_p) - U_n_12_output[1](x1d_m)) / (2*eps)

        dY_ref = -A_mem * q_mem * np.sin(q_mem * x)   # d/dx[A*cos(q*x)]

        tx = 1.0 + dU0dx
        ty = dY_ref + dU1dx
        nrm = np.sqrt(tx**2 + ty**2)

        # tangent (pointing in +x direction)
        t_x_v.append(tx / nrm)
        t_y_v.append(ty / nrm)

        # outward normal of fluid at TOP = rotate tangent CCW = (-ty, tx)/norm
        # points UPWARD (out of fluid, into membrane)
        n_x_v.append(-ty / nrm)
        n_y_v.append( tx / nrm)

    sig_xx_v = np.array(sig_xx_v)
    sig_xy_v = np.array(sig_xy_v)
    sig_yy_v = np.array(sig_yy_v)
    n_x_v    = np.array(n_x_v)
    n_y_v    = np.array(n_y_v)
    t_x_v    = np.array(t_x_v)
    t_y_v    = np.array(t_y_v)

    # 4f. compute sig_nn, sig_nt, f_y exactly as in stokes_FEM.py
    sig_nn_v = (sig_xx_v*n_x_v**2
              + 2*sig_xy_v*n_x_v*n_y_v
              + sig_yy_v*n_y_v**2)

    sig_nt_v = (sig_xx_v*n_x_v*t_x_v
              + sig_xy_v*(n_x_v*t_y_v + n_y_v*t_x_v)
              + sig_yy_v*n_y_v*t_y_v)

    f_y_v    = sig_xy_v*n_x_v + sig_yy_v*n_y_v   # normal traction y-component

    print("sig_nn: min=%.4e  max=%.4e" % (sig_nn_v.min(), sig_nn_v.max()))
    print("sig_nt: min=%.4e  max=%.4e" % (sig_nt_v.min(), sig_nt_v.max()))
    print("f_y:    min=%.4e  max=%.4e" % (f_y_v.min(),    f_y_v.max()))

    # 4g. save CSV per timestep (like stokes_FEM.py surface_stresses%04d.csv)
    outputDict = {
        'x':       mem_x_sorted,
        'y':       np.full_like(mem_x_sorted, h_mem),
        'sig_xx':  sig_xx_v,
        'sig_xy':  sig_xy_v,
        'sig_yy':  sig_yy_v,
        'sig_nn':  sig_nn_v,
        'sig_nt':  sig_nt_v,
        'f_y':     f_y_v,
        'n_x':     n_x_v,
        'n_y':     n_y_v,
        't_x':     t_x_v,
        't_y':     t_y_v,
    }
    # save CSV using numpy (no pandas needed)
    header = ','.join(outputDict.keys())
    data   = np.column_stack(list(outputDict.values()))
    np.savetxt(csv_dir + 'surface_stresses_%04d.csv' % step,
               data, delimiter=',', header=header, comments='')

    # 4h. write XDMF for ParaView (stress and traction on bulk mesh)
    xdmf_stress.write(stress_bulk, t)
    xdmf_stress.write(trac_bulk,   t)

    # ------------------------------------------------------------------
    # Step 5: diagnostics
    # ------------------------------------------------------------------
    pr_bc.print_bcs()

    print("U min =", U_n_12_output.vector().min())
    print("U max =", U_n_12_output.vector().max())
    print("w min =", w_n_output.vector().min())
    print("w max =", w_n_output.vector().max())

    # ------------------------------------------------------------------
    # Step 6: history update
    # ------------------------------------------------------------------
    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(v_n_output)
    fsp.w_n_1.assign(w_n_output)
    fsp.sigma_n_12.assign(fsp.sigma_n_32 - project(phi_output, fsp.Q_phi))
    fsp.sigma_n_32.assign(fsp.sigma_n_12)
    fsp.U_n_32.assign(U_n_12_output)

    fsp.u_n_2.assign(fsp.u_n_1)
    fsp.u_n_1.assign(fsp.u_n)
    fsp.u_n.vector()[:] = fsp.u_n_1.vector()[:] + dt * fsp.u_dot_n.vector()[:]
    fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
    fsp.u_dot_n_1.assign(fsp.u_dot_n)

    fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)
    fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
    fsp.v_fl_n_1.assign(fsp.v_fl_n)
    fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

    if step % rpam.parameters['print_out_stride'] == 0:
        pr_sol.print_solution(t, step, dt)

    print(f'\t{100.0 * t / rpam.parameters["T"]:.1f} %', flush=True)

xdmf_stress.close()
