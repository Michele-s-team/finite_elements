from dolfin import *

mesh = Mesh()
xdmf = XDMFFile("mesh/triangle_mesh.xdmf")

try:
    xdmf.read(mesh)
    print("Mesh successfully read!")
except RuntimeError as e:
    print(f"Error: {e}")