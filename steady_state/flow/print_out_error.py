# this module prints out the error norm between the field obtained from the FE solution and the respective '_0' fields

from fenics import *
import importlib
import os
import pandas as pd
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import function as fu
import function_spaces as fsp
import input_output as io
import print_out_solution as prout
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)


errors = pd.DataFrame({
    'num_cells_mesh': [rmsh.lmsh.mesh.num_cells()],
    'v': [fu.error_norm(project(sqrt((prout.v_output[i] - fsp.v_0[i]) * geo.g(prout.omega_output)[i, j] * (prout.v_output[j] - fsp.v_0[j])), fsp.Q_z), project(Constant(0), fsp.Q_z), rmsh.dx)], 
    'w': [fu.error_norm(prout.w_output, fsp.w_0, rmsh.dx)],
    'z': [fu.error_norm(prout.z_output, fsp.z_0, rmsh.dx)],
    'sigma': [fu.error_norm(prout.sigma_output, fsp.sigma_0, rmsh.dx)],
    'omega': [fu.error_norm(project(sqrt((prout.omega_output[i] - fsp.omega_0[i]) * geo.g_c(prout.omega_output)[i, j] * (prout.omega_output[j] - fsp.omega_0[j])), fsp.Q_z), project(Constant(0), fsp.Q_z), rmsh.dx)], 
    'mu': [fu.error_norm(prout.mu_output, fsp.mu_0, rmsh.dx)],
    })
errors.to_csv(os.path.join(rarg.args.output_directory, 'errors.csv'), index=False, float_format='%.3e')

