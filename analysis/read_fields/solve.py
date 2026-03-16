'''
This code reads a scalar, vector or tensor from a csv file and outputs it as csv, xdmf and h5 file
Run with
    clear; clear; python3 run.py [path of mesh] [path of file to read] [path of file to write]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; INPUT_PATH="/home/fenics/shared/analysis/read_fields/input"; SOLUTION_PATH="/home/fenics/shared/analysis/read_fields/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $INPUT_PATH $SOLUTION_PATH
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
import mesh.load as lmsh
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

fu.read_from_file(io.add_trailing_slash(rarg.args.solution_in_directory) + 'u.csv', fsp.u)
fu.read_from_file(io.add_trailing_slash(rarg.args.solution_in_directory) + 'v.csv', fsp.v)


io.full_print(fsp.u, 'u',
              rarg.args.output_directory,
              rarg.args.output_directory,
              rarg.args.output_directory,
              rarg.args.output_directory)
io.full_print(fsp.v, 'v',
              rarg.args.output_directory,
              rarg.args.output_directory,
              rarg.args.output_directory,
              rarg.args.output_directory)

print('... done.')
