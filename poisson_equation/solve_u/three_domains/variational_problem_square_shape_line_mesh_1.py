from fenics import *
import importlib
import numpy as np
import switch_problem as swi
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# subtract a constant to u_0_1_on_1 in such a way that it gives 0 when integrated over ds_sub_mesh[0][1]["ds_shape"])
print(f'intergral before = {assemble(fsp.u_0_1_on_1 * rmsh.ds_mesh[1]["ds"])}')

mean = assemble(fsp.u_0_1_on_1 * rmsh.ds_mesh[1]['ds']) / assemble(Constant(1) * rmsh.ds_mesh[1]['ds'])
fsp.u_0_1_on_1.assign(project(fsp.u_0_1_on_1 - mean, fsp.Q[1]))

print(f'integral after = {assemble(fsp.u_0_1_on_1 * rmsh.ds_mesh[1]["ds"])}')



# impose a Dirichlet boundary condition on the left vertex in order to set C[1] -> 0
bc_l = DirichletBC(fsp.Q[1], Constant(0), rmsh.mf[1], rmsh.lmsh.mesh_parameters[1]['vertex_l_id'])

bcs = [bc_l]

# variational functional
F = (fsp.u[1].dx(0) - fsp.u_0_1_on_1) *  fsp.nu_u[1].dx(0) * rmsh.dx_mesh[1]
