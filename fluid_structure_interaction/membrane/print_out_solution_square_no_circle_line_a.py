from fenics import *


import csv
import function as fu
import function_spaces as fsp
import importlib
import input_output as io
import mesh.utils as msh
import os
import solution_paths as solpath
import ufl

import differential_geometry.manifold.geometry as geo
import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
import parameters.read.solution as rpam
import physics.fluid_mechanics as flu
import runtime_arguments as rarg
import switch_problem as swi

cu = importlib.import_module(swi.cu)
fi = importlib.import_module(swi.fi)
rmsh = importlib.import_module(swi.rmsh)



i, j, k = ufl.indices( 3 )


def print_solution(t, step, dt):

    # WARNING: here `var_tensor_sigma_fl` uses `u_n` while in the variational problem it is defined with u_n_1 - start 
    fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n, fsp.sigma_fl_n_12, fsp.u_n, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
    # WARNING: here `var_tensor_sigma_fl` uses `u_n` while in the variational problem it is defined with u_n_1 - end
    fu.transfer_2d_to_1d_curve(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rarg.args.input_directory)

    # 1. write fields 

    # 1.1 membrane problem
    v_bar_dummy, w_bar_dummy, phi_dummy, v_n_dummy, w_n_dummy, U_n_12_dummy, nu_n_12_dummy, psi_n_12_dummy, mu_n_12_dummy = fsp.psi_mem.split( deepcopy=True )

    # 1.1.1 physical fields

    # print solution to file
    # append to the full time series solution at the current t
    fi.xdmffile_v_bar.write( v_bar_dummy, t )
    fi.xdmffile_w_bar.write( w_bar_dummy, t )
    fi.xdmffile_v_n.write( v_n_dummy, t )
    fi.xdmffile_w_n.write( w_n_dummy, t )
    fi.xdmffile_sigma_n_12.write( fsp.sigma_n_12, t - dt / 2.0 )
    fi.xdmffile_phi.write( phi_dummy, t )
    fi.xdmffile_u_n_12.write( U_n_12_dummy, t - dt / 2.0 )
    fi.xdmffile_nu_n_12.write( nu_n_12_dummy, t - dt / 2.0 )
    fi.xdmffile_psi_n_12.write( psi_n_12_dummy, t - dt / 2.0 )
    fi.xdmffile_mu_n_12.write( mu_n_12_dummy, t - dt / 2.0 )

    io.full_print(v_bar_dummy, 'v_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(w_bar_dummy, 'w_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(v_n_dummy, 'v_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(w_n_dummy, 'w_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(U_n_12_dummy, 'U_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(nu_n_12_dummy, 'nu_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(psi_n_12_dummy, 'psi_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(mu_n_12_dummy, 'mu_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    
    io.full_print(fsp.X_ref, 'X_ref_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    #1.1.2 post-processing fields

    #1.1.2.1 forces
    io.full_print(fsp.f_el_n_12, 'f_el_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(project(2 * fsp.sigma_n_12 * fsp.mu_n_12, fsp.Q_phi), 'f_lap_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(project(
            2.0 * rpam.parameters['eta'] * (geo.g_c( psi_n_12_dummy, nu_n_12_dummy )[i, k] * geo.Nabla_v( v_n_dummy, psi_n_12_dummy, nu_n_12_dummy )[j, k] *
                                                 (geo.b( psi_n_12_dummy, nu_n_12_dummy ))[i, j] - 2.0 * w_n_dummy * (
                                                         2.0 * ((mu_n_12_dummy) ** 2) - geo.K( psi_n_12_dummy, nu_n_12_dummy ))), fsp.Q_phi), 'f_vis_n_n_' + str(step + 1), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(project(
            geo.from_3D_to_normal(psi_n_12_dummy, 
                flu.dFdl(
                        fsp.var_tensor_sigma_fl_on_mem, 
                        geo_al.normal(psi_n_12_dummy, nu_n_12_dummy)
                        ), 
                nu_n_12_dummy), fsp.Q_phi), 'f_fl_n_12_' + str(step + 1), \
    solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    # 1.1.2.2 intertial terms
    io.full_print(project(
        rpam.parameters['rho'] * ( v_n_dummy[i] * v_n_dummy[k] * geo.b( psi_n_12_dummy, nu_n_12_dummy )[k, i] ), fsp.Q_phi), 'f_ine_v_n_' + str(step + 1), \
    solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    io.full_print(project(
        rpam.parameters['rho'] * ( v_n_dummy[i] * (w_n_dummy.dx(i)) ), fsp.Q_phi), 'f_ine_w_n_' + str(step + 1), \
    solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


    # 1.2 mesh problem

    io.full_print(fsp.u_n, 'u_n_' + str(step + 1), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.u_dot_n, 'u_dot_n_' + str(step + 1), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    # include the snapshot in xdmf files
    fi.xdmffile_u_n.write(fsp.u_n, t)
    fi.xdmffile_u_dot_n.write(fsp.u_dot_n, t)

    # 1.3 fluid problem
    io.full_print(fsp.v_fl_bar, 'v_fl_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.v_fl_n, 'v_fl_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.sigma_fl_n_12, 'sigma_fl_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.phi_fl, 'phi_fl_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


    # 2. include snapshots in xdmf files
    fi.xdmffile_v_fl_n.write(fsp.v_fl_n, t)
    fi.xdmffile_v_fl_bar.write(fsp.v_fl_bar, t)
    fi.xdmffile_sigma_fl.write(fsp.sigma_fl_n_12, t - dt / 2.0)
    fi.xdmffile_phi_fl.write(fsp.phi_fl, t)

    io.full_print_deformed(fsp.v_fl_bar, fsp.u_n, 'v_fl_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.v_fl_n, fsp.u_n, 'v_fl_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.sigma_fl_n_12, fsp.u_n, 'sigma_fl_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.phi_fl, fsp.u_n, 'phi_fl_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


    # 3. write the curve

    tab_cspline = [
        [
            float(cu.cspline[0](cu.arc_length_tab[-1] * alpha/(rpam.parameters['n_points_output_cspline']-1))), 
            float(cu.cspline[1](cu.arc_length_tab[-1] * alpha/(rpam.parameters['n_points_output_cspline']-1)))
            ] 
        for alpha in range(rpam.parameters['n_points_output_cspline']) 
        ]

    with open(os.path.join(solpath.snapshots_csv_path, f'cspline_n_{step + 1}.csv'), 'w', newline='') as cspline_file:

        writer = csv.writer(cspline_file)
        writer.writerow([':0',':1'])  
        writer.writerows(tab_cspline) 


    #4. Write the deformed mesh to file

    deformed_mesh = msh.deform_mesh(rmsh.lmsh.mesh[0], fsp.u_n)

    with XDMFFile(os.path.join(solpath.snapshots_path, 'mesh_n_' + str(step + 1) + '.xdmf')) as xdmf:
        xdmf.write(deformed_mesh)

    io.print_mesh_vertices_to_csv(deformed_mesh, os.path.join(solpath.snapshots_csv_path, 'vertex_mesh_n_' + str(step + 1) + '.csv'))
    io.print_mesh_lines_to_csv(deformed_mesh, os.path.join(solpath.snapshots_csv_path, 'line_mesh_n_' + str(step + 1) + '.csv'))
      

    #5. write shape vertices 

    input_path = os.path.join(rarg.args.input_directory, f"mesh_0/boundary_points_id_{rmsh.parameters['mesh_1_id']}.csv")
    output_path = os.path.join(rarg.args.output_directory, f"snapshots/csv/boundary_points_id_{rmsh.parameters['mesh_1_id']}_n_{step + 1}.csv")
    os.system(f'cp {input_path} {output_path}')


