import sys

module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import differential_geometry.boundary.geometry as bgeo

import switch_problem as swi
import solution_paths as solpath
import runtime_arguments as rarg
import mesh.load as lmsh
import input_output as sys_io
from fenics import *
import importlib
import ufl as ufl

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)



def print_solution(step):
    
    sys_io.full_print(fsp.u_n, 'u_n_' + str(step + 1), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)




