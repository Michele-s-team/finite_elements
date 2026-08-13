import csv
from fenics import *
import importlib
import os

import differential_geometry.boundary.geometry as bgeo
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
sh = importlib.import_module(swi.sh)
vp = importlib.import_module(swi.vp)




def print_solution(t, step, dt):

    #1 unpack the mixed field 

    v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy, c_n_dummy, mu_n_dummy, grad_u_n_dummy = fsp.psi.split( deepcopy=True )

    #2 write to xdmf files

    fi.xdmffile_v_n.write(v_n_dummy, t)
    fi.xdmffile_sigma_n.write(sigma_n_dummy, t)

    fi.xdmffile_u_n.write(u_n_dummy, t)
    fi.xdmffile_u_dot_n.write(u_dot_n_dummy, t)

    fi.xdmffile_c_n.write(c_n_dummy, t)

    fi.xdmffile_mu_n.write(mu_n_dummy, t)
    fi.xdmffile_grad_u_n.write(grad_u_n_dummy, t)

    # 3 write snapshots
    

    # 3.1 reference configuration and deformation fields
    io.full_print(v_n_dummy, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])    
    io.full_print(sigma_n_dummy, 'sigma_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

    
    io.full_print(u_n_dummy, 'u_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    io.full_print(u_dot_n_dummy, 'u_dot_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    
    io.full_print(c_n_dummy, 'c_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    
    io.full_print(mu_n_dummy, 'mu_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    io.full_print(grad_u_n_dummy, 'grad_u_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

    
    # 3.1.1 write additional fields 

    # 3.1.1.1 write f_shape
    io.full_print(
        project(
            vp.f_shape(
                        c_n_dummy, 
                        u_n_dummy, 
                        mu_n_dummy, 
                        bgeo.field_facet_normal_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](vp.sub_mesh_0_label), rmsh.ds_mesh[0]['dS_shape'], interior=True)
            ), fsp.Q_u_n), 
        'f_shape_n_' + str(step),
        solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0]
    )

    # 3.1.1.2 write the cspline

    tab_cspline = [[float(sh.cspline[0](alpha/rpam.parameters['n_points_output_cspline'])), float(sh.cspline[1](alpha/rpam.parameters['n_points_output_cspline']))] for alpha in range(rpam.parameters['n_points_output_cspline']) ]

    with open(os.path.join(solpath.snapshots_csv_path, f'cspline_n_{step}.csv'), 'w', newline='') as cspline_file:
        writer = csv.writer(cspline_file)
        writer.writerow([':0',':1'])          # header
        writer.writerows(tab_cspline)        # one point per row

    
    # 3.2.1 deformed with u_n
    
    io.full_print_deformed(v_n_dummy, u_n_dummy, 'v_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    io.full_print_deformed(sigma_n_dummy, u_n_dummy, 'sigma_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

    io.full_print_deformed(c_n_dummy, u_n_dummy, 'c_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    
    io.full_print_deformed(mu_n_dummy, u_n_dummy, 'mu_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    
    # 3.2.2 deformed with u_0

    io.full_print_deformed(v_n_dummy, fsp.u_0, 'v_0_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    
    io.full_print_deformed(sigma_n_dummy, fsp.u_0, 'sigma_0_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

    # 3.5 phi_0
    io.full_print(fsp.phi_0, 'phi_0_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    
    # 3.6 u_0
    io.full_print(fsp.u_0, 'u_0_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
 


    #4. Write the deformed mesh to file

    # 4.1 write the mesh deformed according to u_n
    deformed_mesh = msh.deform_mesh(rmsh.lmsh.mesh[0], u_n_dummy)

    with XDMFFile(solpath.snapshots_path + 'mesh_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)

    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_n_' + str(step) + '.csv')


    # 4.2 write the mesh deformed according to u_0
    deformed_mesh_0 = msh.deform_mesh(rmsh.lmsh.mesh[0], fsp.u_0)

    with XDMFFile(solpath.snapshots_path + 'mesh_0_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh_0)

    io.print_mesh_vertices_to_csv(deformed_mesh_0, solpath.snapshots_csv_path + 'vertex_mesh_0_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh_0, solpath.snapshots_csv_path + 'line_mesh_0_n_' + str(step) + '.csv')


    # 5 write shape vertices 
    input_path = os.path.join(rarg.args.input_directory, f"mesh_0/boundary_points_id_{rmsh.parameters['shape_id']}.csv")
    output_path = os.path.join(rarg.args.output_directory, f"snapshots/csv/boundary_points_id_{rmsh.parameters['shape_id']}_n_{step}.csv")
    os.system(f'cp {input_path} {output_path}')

