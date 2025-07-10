import files as fi
import os
from fenics import assemble, dot, sqrt, File, dx, XDMFFile
import runtime_arguments as rarg
import function_spaces_steady as fsp
import input_output as io
import load_mesh as lmsh
import solution_paths as solpath


def print_solution(t, step, dt):
    # include the snapshot in xdmf files
    fi.xdmffile_v.write(fsp.v_n, t)
    fi.xdmffile_v_.write(fsp.v_, t)
    fi.xdmffile_sigma.write(fsp.sigma_n_12, t - dt / 2.0)
    fi.xdmffile_phi.write(fsp.phi, t)

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
    io.full_print(fsp.phi, 'phi_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'scalar')
    
def print_solution_steady(u, p=None):
    """
    Solution for u
    """
    from fenics import assemble, dot, sqrt, File, dx
    import os
    import runtime_arguments as rarg

    out_dir = rarg.args.output_directory
    os.makedirs(out_dir, exist_ok=True)

    # Norms for u
    u_l2  = sqrt( assemble(dot(u, u) * dx) )
    u_inf = u.vector().norm('linf')
    print(f"||u||_L2 = {u_l2:.6g}")
    print(f"||u||_L∞ = {u_inf:.6g}")

    # only use p if it is not NOne
    if p is not None:
        p_l2 = sqrt( assemble(p**2 * dx) )
        print(f"||p||_L2 = {p_l2:.6g}")
        File(os.path.join(out_dir, 'p_steady.pvd')) << p

    # save u
    File(os.path.join(out_dir, 'u_steady.pvd')) << u
    with XDMFFile(os.path.join(out_dir, 'u_steady.xdmf')) as xdmf_u:
        xdmf_u.write(u)

    if p is not None:
        File(os.path.join(out_dir, 'p_steady.pvd')) << p
        with XDMFFile(os.path.join(out_dir, 'p_steady.xdmf')) as xdmf_p:
            xdmf_p.write(p)
    

