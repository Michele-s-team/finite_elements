"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
Run with
    python solve.py [problem name] [path where to read the mesh] [path where to store the solution]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_flat_icps/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $SOLUTION_PATH
"""

import argparse
import colorama as col
import dolfin
from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import files as fi
import function_spaces as fsp
import runtime_arguments as rarg
import switch_problem as swi
import print_out_solution as pr_sol

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
pr_bc = importlib.import_module(swi.prout_bc)

print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)
print(f"Radius of mesh cell = {col.Fore.BLUE}{rmsh.r_mesh}{col.Style.RESET_ALL}")

# import calculus as calc

# parser = argparse.ArgumentParser()
# parser.add_argument("input_directory")
# parser.add_argument("output_directory")
# args = parser.parse_args()

# L = 2.2
# h = 0.41
# r = 0.05
# c_r = [0.2, 0.2]

# # read mesh
# mesh = Mesh()
# with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
#     infile.read(mesh)
# mvc = MeshValueCollection("size_t", mesh, 2)
# with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
#     infile.read(mvc, "name_to_read")


# Define boundaries
# inflow = 'near(x[0], 0)'
# outflow = f'near(x[0], {L})'
# walls = f'near(x[1], 0) || near(x[1], {h})'
# cylinder = f'on_boundary && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) < {(r + calc.min_dist_c_r_rectangle(L, h, c_r)) / 2}'


# Define expressions used in variational forms
# n = FacetNormal(mesh)
# k = Constant(dt)
# mu = Constant(mu)
# rho = Constant(rho)


# Time-stepping
t = 0
for n in range(vp.num_steps):
    # Update current time
    t += vp.dt

    # Step 1: Tentative velocity step
    b1 = assemble(vp.L1)
    [bc.apply(b1) for bc in vp.bc_u]
    solve(vp.A1, fsp.u_bar.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(vp.L2)
    [bc.apply(b2) for bc in vp.bc_p]
    solve(vp.A2, fsp.p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(vp.L3)
    solve(vp.A3, fsp.u_bar.vector(), b3, 'cg', 'sor')

    # Save solution to file (XDMF/HDF5)
    fi.xdmffile_u.write(fsp.u_bar, t)
    fi.xdmffile_p.write(fsp.p_, t)

    # Update previous solution
    fsp.u_n.assign(fsp.u_bar)
    fsp.p_n.assign(fsp.p_)

    print("\t%.2f %%" % (100.0 * (t / vp.T)), flush=True)


fi.xdmffile_u.close()
fi.xdmffile_p.close()