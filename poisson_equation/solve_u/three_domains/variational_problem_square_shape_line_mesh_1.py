from fenics import *
import importlib
import numpy as np
import switch_problem as swi
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# impose a Dirichlet boundary condition on the left vertex in order to set C[1] -> 0
bc_l = DirichletBC(fsp.Q[1], Constant(0), rmsh.mf[1], rmsh.lmsh.mesh_parameters[1]['vertex_l_id'])

bcs = [bc_l]

# variational functional
F = (fsp.u[1].dx(0) - fsp.u_0_1_on_1) *  fsp.nu_u[1].dx(0) * rmsh.dx_mesh[1]
