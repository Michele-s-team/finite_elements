import dolfin
from fenics import *

import mesh as msh
import runtime_arguments as rarg


#read the mesh
mesh = msh.read_mesh(rarg.args.input_directory + "/triangle_mesh.xdmf")