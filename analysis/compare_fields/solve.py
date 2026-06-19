'''
This code reads a pair of scalar, vector or tensor fields from a csv file and computes their difference by using error_norm
Run with
    clear; clear; python3 run.py [path of mesh] [path of file to read] [path of file to write]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; INPUT_PATH="/home/fenics/shared/analysis/compare_fields/input"; SOLUTION_PATH="/home/fenics/shared/analysis/compare_fields/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $INPUT_PATH $SOLUTION_PATH
'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import function as fu
import input_output as io
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

fu.read_from_file(io.add_trailing_slash(rarg.args.solution_in_directory) + 'z_n_12_1250.csv', fsp.u)
fu.read_from_file(io.add_trailing_slash(rarg.args.solution_in_directory) + 'z_n_12_2500.csv', fsp.v)


print(f' error_norm = {fu.error_norm(fsp.u, fsp.v, rmsh.dx)}')


