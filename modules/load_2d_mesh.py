import dolfin
from fenics import *

import command as cmd
import mesh as msh
import runtime_arguments as rarg

if cmd.check_if_file_exists(rarg.args.input_directory + "/tetrahedron_mesh.xdmf"):
    mesh = msh.read_mesh(rarg.args.input_directory + "/tetrahedron_mesh.xdmf")
    print('3d mesh')
else:
    if cmd.check_if_file_exists(rarg.args.input_directory + "/triangle_mesh.xdmf"):
        mesh = msh.read_mesh(rarg.args.input_directory + "/triangle_mesh.xdmf")
        print('2d mesh')
    else:
        if cmd.check_if_file_exists(rarg.args.input_directory + "/line_mesh.xdmf"):
            mesh = msh.read_mesh(rarg.args.input_directory + "/line_mesh.xdmf")
            print('1d mesh')
