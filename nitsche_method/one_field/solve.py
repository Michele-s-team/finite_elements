'''
This code solves the Poisson equation with Dirichlet BCs

u = u_D on \partial \Omega

by imposing the BCs with Nitsche's method. Run with  

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/nitsche_method/one_field/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_slice $MESH_PATH $SOLUTION_PATH
'''

from fenics import *
from mshr import *
import ufl as ufl

i, j, k, l = ufl.indices(4)

# parser = argparse.ArgumentParser()
# args = parser.parse_args()



# Create mesh
# channel = Rectangle(Point(0, 0), Point(1.0, 1.0))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# domain = channel
# alpha = Constant(10.0)
# mesh = generate_mesh(domain, 16)


solve(F == 0, u)


