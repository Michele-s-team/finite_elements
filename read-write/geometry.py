import math

from fenics import *
from mshr import *
import numpy as np
# from dolfin import *
import meshio
import ufl as ufl
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("mesh_old_directory")
parser.add_argument("solution_old_directory")
parser.add_argument("solution_new_directory")
parser.add_argument("N")
parser.add_argument("i")
args = parser.parse_args()

#CHANGE PARAMETERS HERE
L = 2.2
h = L
r = 0.05
c_r = [L/2.0, h/2.0]
N = (int)(args.N)
increment = (int)(args.i)
# time step size
#CHANGE PARAMETERS HERE



#read mesh
mesh=Mesh()
with XDMFFile((args.mesh_old_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.mesh_old_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")


#this is the facet normal vector, which cannot be plotted as a field
#n_overline = \overline{n}_notes_{on the circle}
n_overline = FacetNormal(mesh)

# Define function spaces
#finite elements for sigma .... omega
P_v_bar = VectorElement( 'P', triangle, 2 )
P_w_bar = FiniteElement( 'P', triangle, 1 )
P_phi = FiniteElement('P', triangle, 2)
P_v_n = VectorElement( 'P', triangle, 2 )
P_w_n = FiniteElement( 'P', triangle, 1 )
P_omega_n = VectorElement( 'P', triangle, 3 )
P_z_n = FiniteElement( 'P', triangle, 1 )

element = MixedElement( [P_v_bar, P_w_bar, P_phi, P_v_n, P_w_n, P_omega_n, P_z_n] )
#total function space
Q = FunctionSpace(mesh, element)
#function spaces for vbar .... zn
Q_v_bar = Q.sub(0).collapse()
Q_w_bar = Q.sub(1).collapse()
Q_phi = Q.sub(2).collapse()
Q_v_n = Q.sub(3).collapse()
Q_w_n = Q.sub(4).collapse()
Q_omega_n = Q.sub(5).collapse()
Q_z_n= Q.sub(6).collapse()


# norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l  = 'near(x[0], 0.0)'
boundary_r  = 'near(x[0], 2.2)'
boundary_lr  = 'near(x[0], 0) || near(x[0], 2.2)'
boundary_tb  = 'near(x[1], 0) || near(x[1], 0.41)'
boundary_square = 'on_boundary && sqrt(pow(x[0] - 2.2/2.0, 2) + pow(x[1] - 0.41/2.0, 2)) > 0.1'
boundary_circle = 'on_boundary && sqrt(pow(x[0] - 2.2/2.0, 2) + pow(x[1] - 0.41/2.0, 2)) < 0.1'
#CHANGE PARAMETERS HERE