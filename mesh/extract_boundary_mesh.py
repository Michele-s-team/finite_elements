import h5py
import meshio
import numpy as np
from dolfin import *

dim = 3
N = 2**5
mesh = UnitCubeMesh(N, N, N)

with XDMFFile("cube_mesh.xdmf") as xdmf:
    xdmf.write(mesh)

bdim = dim-1
bmesh = BoundaryMesh(mesh, "exterior")
mapping = bmesh.entity_map(bdim)
part_of_bot = MeshFunction("size_t", bmesh, bdim)
for cell in cells(bmesh):
    curr_facet_normal = Facet(mesh, mapping[cell.index()]).normal()
    if near(curr_facet_normal.y(), -1.0):  # On bot boundary
        part_of_bot[cell] = 1

bot_boundary = SubMesh(bmesh, part_of_bot, 1)
#File('bot_boundary.pvd') << bot_boundary
with XDMFFile("bot_mesh.xdmf") as xdmf:
    xdmf.write(bot_boundary)


in_mesh = meshio.read("bot_mesh.xdmf")

cells = in_mesh.get_cells_type("triangle")
points = np.delete(in_mesh.points, 1, axis=1)
out_mesh = meshio.Mesh(points=points, cells={"triangle": cells})
meshio.write("pruned_mesh.xdmf", out_mesh)


mesh2D = Mesh()
with XDMFFile("pruned_mesh.xdmf") as xdmf:
    xdmf.read(mesh2D)
print(mesh2D.geometry().dim())