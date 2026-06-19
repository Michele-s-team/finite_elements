from fenics import *

import mesh.load as lmsh
import parameters.read.analysis as rpam


Q = FunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'])

u = Function(Q)
v = Function(Q)

