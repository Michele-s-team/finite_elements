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
    MESH_PATH="/home/fenics/shared/generate_mesh/3d/box_ball/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_flat_icps/solution"; rm -rf $SOLUTION_PATH; python3 solve.py box_ball $MESH_PATH $SOLUTION_PATH

For box_ball problem one gets turbulent behavior with
    L =  [2.2, 0.41, 0.41]
    mu =  0.0001
    T =  1
    N =  10000
    /home/fenics/shared/generate_mesh/3d/box_ball#     SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_box_ball_mesh.py 0.05 $SOLUTION_PATH
    /home/fenics/shared/dynamics/channel_with_cylinder_flat_icps#     MESH_PATH="/home/fenics/shared/generate_mesh/3d/box_ball/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_flat_icps/solution"; rm -rf $SOLUTION_PATH; python3 solve.py box_ball $MESH_PATH $SOLUTION_PATH
"""

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

# Time-stepping
t = 0
step = 0

for n in range(vp.num_steps):
    # Update current time
    t += vp.dt
    step += 1

    # Step 1: Tentative velocity step
    b1 = assemble(vp.L1)
    [bc.apply(b1) for bc in vp.bc_u_bar]
    solve(vp.A1, fsp.u_bar.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(vp.L2)
    [bc.apply(b2) for bc in vp.bc_p]
    solve(vp.A2, fsp.p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(vp.L3)
    solve(vp.A3, fsp.u_n.vector(), b3, 'cg', 'sor')

    pr_bc.print_bcs()

    # Update previous solution
    fsp.p_n.assign(fsp.p_)

    pr_sol.print_solution(t, step, vp.dt)

    print("\t%.2f %%" % (100.0 * (t / vp.T)), flush=True)

fi.xdmffile_u_bar.close()
fi.xdmffile_u_n.close()
fi.xdmffile_p_n.close()
