'''
Notation:
- sub_mesh: either of the parts of the total mesh

- sf_sub_mesh: a list of map functions, where sf_sub_mesh[i] is the map function for the triangles of the i-th sub_mesh
- mf_sub_mesh: a list of map functions, where mf_sub_mesh[i] is the map function for the lines of the i-th sub_mesh
'''

from fenics import *
import numpy as np

import input_output as io
import load_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")

# read the lines
cf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), io.add_trailing_slash(rarg.args.input_directory) + "line_mesh.h5", "cf")
# read the vertices
vf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, io.add_trailing_slash(rarg.args.input_directory) + "vertex_mesh.h5", "vf")

'''
# Add this debug code right after the parameters line
print(f"DEBUG: Manually creating submesh for line ID 2...")
try:
    manual_line_submesh = SubMesh(lmsh.mesh, mf, 2)
    print(f"DEBUG: Manual line submesh - vertices: {manual_line_submesh.num_vertices()}, cells: {manual_line_submesh.num_cells()}")
except Exception as e:
    print(f"DEBUG: Error creating manual line submesh: {e}")

print(f"DEBUG: Manually creating submesh for surface ID 1...")
try:
    manual_surface_submesh = SubMesh(lmsh.mesh, sf, 1)
    print(f"DEBUG: Manual surface submesh - vertices: {manual_surface_submesh.num_vertices()}, cells: {manual_surface_submesh.num_cells()}")
except Exception as e:
    print(f"DEBUG: Error creating manual surface submesh: {e}")


# DEBUG: Check what marker IDs are actually in the mesh functions
# Add this right after reading the mesh functions
print(f"DEBUG: Unique surface marker IDs in sf: {np.unique(sf.array())}")
print(f"DEBUG: Unique line marker IDs in mf: {np.unique(mf.array())}")
print(f"DEBUG: Parameters - sub_mesh_0_id: {parameters.get('sub_mesh_0_id', 'NOT FOUND')}")
print(f"DEBUG: Parameters - sub_mesh_1_id: {parameters.get('sub_mesh_1_id', 'NOT FOUND')}")

print(f'sub_meshes = {lmsh.sub_meshes}')

for i, sub_mesh in enumerate(lmsh.sub_meshes):
    print(f"Submesh {i}:")
    for d in range(sub_mesh.topology().dim() + 1):
        print(f"  dim {d} entities: {sub_mesh.num_entities(d)}")

    # DEBUG: Additional info about empty submeshes
    if sub_mesh.num_vertices() == 0:
        print(f"  DEBUG: Submesh {i} is EMPTY!")



# create a list of map functions for triangles and lines for each sub_mesh
sf_sub_mesh = []
mf_sub_mesh = []
for sub_mesh in lmsh.sub_meshes:
    sf_sub_mesh.append(msh.transfer_cell_tags_to_sub_mesh(sub_mesh, sf))
    mf_sub_mesh.append(msh.transfer_facet_tags_to_sub_mesh(lmsh.mesh, sub_mesh, mf))

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

# create line and surface elements for sub_meshes
dx_sub_mesh = []

dx_sub_mesh.append(Measure("dx", domain=lmsh.sub_meshes[0], subdomain_data=sf_sub_mesh[0], subdomain_id=parameters[f"sub_mesh_{0}_id"]))
dx_sub_mesh.append(Measure("dx", domain=lmsh.sub_meshes[1]))
'''
