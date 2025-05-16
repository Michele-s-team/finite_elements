'''
This code solves the Poisson equation with Dirichlet BCs

u = u_D on \partial \Omega

by imposing the BCs with Nitsche's method. Run with  

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/nitsche_method/one_field/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH
'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


# import ufl as ufl

# i, j, k, l = ufl.indices(4)

# parser = argparse.ArgumentParser()
# args = parser.parse_args()



# Create mesh
# channel = Rectangle(Point(0, 0), Point(1.0, 1.0))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# domain = channel
# mesh = generate_mesh(domain, 16)


solve(vp.F == 0, fsp.u)

prout_bc = importlib.import_module(swi.prout_bc)



