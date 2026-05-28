import files as fi
import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import solution_paths as solpath


def print_solution(t, step):

    v_n_dummy, sigma_n_dummy = fsp.psi.split( deepcopy=True )


    # include the snapshot in xdmf files
    fi.xdmffile_v.write(v_n_dummy, t)
    fi.xdmffile_sigma.write(sigma_n_dummy, t)

    # print the snapshot in a separate file
    io.full_print(v_n_dummy, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(sigma_n_dummy, 'sigma_n' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
