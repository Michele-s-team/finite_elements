import colorama as col
from fenics import *
import importlib
import os

import calculus as cal
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import numpy as np
import runtime_arguments as rarg

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
mf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, rarg.args.input_directory + "/line_mesh.xdmf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

print(f'Mesh quality = {msh.custom_mesh_quality(lmsh.mesh)}')

parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, "mesh_metadata.csv"))

print(f"Radius of mesh cell = {col.Fore.BLUE}{r_mesh}{col.Style.RESET_ALL}")


polygon_coordinates = None
if parameters['polygon_format'] == 'coordinates':
    # the polygon shape is provided directly as a sequence of coordinates of the polygon points -> set polygon_coordinates to these coordinates

    print('The polygon shape is provided as a set of coordinates.')
    polygon_coordinates = parameters['polygon_coordinates']

else:
    #  the polygon shape is a given, parameteric geometrical shape, and it is provided in terms of the parameters of this shape

    if parameters['polygon_format'] == 'ellipse':
    # the polygon shape is an ellipse -> obtain polygon_coordinates from the ellipse parameters
        print('The polygon shape is an ellipse.')
        polygon_coordinates = cal.points_ellipse(parameters['a'], parameters['b'], parameters['c'], parameters['N'],
                                                 phi=parameters['phi'])

    # here you can have other cases corresponding to other geometrical shapes (circle, etc... )



# test for surface elements
dx = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters['surface_id'])
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters['line_l_id'])
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters['line_r_id'])
ds_t = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters['line_t_id'])
ds_b = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters['line_b_id'])
ds_poly = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters['polygon_id'])
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds_l_tb_poly = ds_l + ds_t + ds_b + ds_poly
ds = ds_square + ds_poly

check_mesh_module = importlib.import_module('mesh.check_tags.square_polygon')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)