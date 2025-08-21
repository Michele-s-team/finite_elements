from fenics import *
import runtime_arguments as rarg

import input_output as io

parameters = io.read_parameters_from_csv_file("parameters_bc_line.csv")

mesh = IntervalMesh(parameters['N'], parameters['x_min'], parameters['x_max'])

# Tag different regions of the line (1D entities - cells)
cell_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
cell_markers.set_all(1)  # Tag entire line as region 1

dx = Measure("dx", domain=mesh, subdomain_data=cell_markers, subdomain_id=1)

print(f'int = {assemble(Constant(1)*dx)}')