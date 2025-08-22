from fenics import *
import math
import runtime_arguments as rarg

import input_output as io
import read_parameters_generate_mesh

parameters = io.read_parameters_from_csv_file("parameters_bc_line.csv")

mesh = IntervalMesh(parameters['N'], parameters['x_l'], parameters['x_r'])

# create a function for the lines
cf = MeshFunction("size_t", mesh, mesh.topology().dim())
cf.set_all(parameters['line_id'])  # Tag entire line as region parameters['line_id']

# creat a function for the vertices
vf = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
for vertex in vertices(mesh):
    x = vertex.point().x()  # Get x-coordinate

    if math.isclose(x, parameters['x_l']):
        vf[vertex] = parameters['vertex_l_id']

    if math.isclose(x, parameters['x_r']):
        vf[vertex] = parameters['vertex_r_id']


dx = Measure("dx", domain=mesh, subdomain_data=cf, subdomain_id=parameters['line_id'])
ds = Measure("ds", domain=mesh, subdomain_data=vf, subdomain_id=parameters['vertex_l_id'])

print(f'int = {assemble(Constant(1)*dx)}')