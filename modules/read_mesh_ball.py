'''
This code reads the 3d mesh generated from generate_ball_mesh.py and it creates dvs and dss from labelled components of the mesh
'''

import argparse
import dolfin
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

# CHANGE PARAMETERS HERE
r = 1
c_r = [0, 0, 0]

volume_id = 1
surface_id = 2
# CHANGE PARAMETERS HERE

# read the mesh
mesh = Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), (args.input_directory) + "/tetrahedron_mesh.xdmf")
xdmf.read(mesh)

# read the tetrahedra
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile((args.input_directory) + "/tetrahedron_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

# read the triangles
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

boundary_mesh = BoundaryMesh(mesh, "exterior")
with XDMFFile("solution/boundary_mesh.xdmf") as xdmf:
    xdmf.write(boundary_mesh)

'''
#read the lines
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the vertices
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile((args.input_directory) + "/vertex_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()
'''
dv = Measure("dx", domain=mesh, subdomain_data=cf, subdomain_id=volume_id)  # volume measure
ds = Measure("ds", domain=mesh, subdomain_data=sf, subdomain_id=surface_id)  # surface measure
# dS_custom = Measure("dS", domain=mesh, subdomain_data=sf)    # Point measure for points in the mesh


import check_mesh_tags_ball

print(f'Module {__file__} called {check_mesh_tags_ball.__file__}', flush=True)
