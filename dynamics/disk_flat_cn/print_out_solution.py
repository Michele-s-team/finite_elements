import files as fi
import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import solution_paths as solpath


def print_solution(t, step, dt):

    phi_output, omega_output = fsp.phi_omega.split(deepcopy=True)


    # include the snapshot in xdmf files
    fi.xdmffile_v.write(fsp.v_n, t)
    fi.xdmffile_v_.write(fsp.v_, t)
    fi.xdmffile_sigma.write(fsp.sigma_n_12, t - dt / 2.0)
    fi.xdmffile_phi.write(phi_output, t)

    # print the snapshot in a separate file
    io.full_print(fsp.v_, 'v_bar_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'vector')
    io.full_print(fsp.v_n, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'vector')
    
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'scalar')
    
    io.full_print(phi_output, 'phi_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'scalar')
    io.full_print(omega_output, 'omega_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'vector')
