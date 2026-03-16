import files as fi
import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import solution_paths as solpath


def print_solution(t, step, dt):
    fi.xdmffile_u_bar.write(fsp.u_bar, t)
    fi.xdmffile_u_n.write(fsp.u_n, t)
    fi.xdmffile_p_n.write(fsp.p_n, t)

    # print the snapshot in a separate file
    io.full_print(fsp.u_bar, 'u_bar_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'vector')
    io.full_print(fsp.u_n, 'u_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'vector')
    io.full_print(fsp.p_n, 'p_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'scalar')
