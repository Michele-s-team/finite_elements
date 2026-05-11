import importlib

import files as fi
import function_spaces as fsp
import input_output as io
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


def print_solution(t, step, dt):

    v__dummy, phi_dummy, v_n_dummy = fsp.psi.split( deepcopy=True )


    # include the snapshot in xdmf files
    fi.xdmffile_v.write(v_n_dummy, t)
    fi.xdmffile_v_.write(v__dummy, t)
    fi.xdmffile_sigma.write(fsp.sigma_n_12, t - dt / 2.0)
    fi.xdmffile_phi.write(phi_dummy, t)

    # print the snapshot in a separate file
    io.full_print(v__dummy, 'v_bar_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    io.full_print(v_n_dummy, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    
    io.full_print(phi_dummy, 'phi_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
   