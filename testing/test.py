'''
this code performs a test of mlutiple parts of the code by comparing the solution in csv files across two commits

run with

python3 test.py [sha of commit_a] [sha of commit_b]
Example
    clear; python3 test.py 7e670cf2a5ba005ab77202c124d691eaa5bc17ea 0119597915ccfedf8560e5092c2cda8ae74ce152
'''

import colorama as col
import os
import sys


#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import runtime_arguments as rarg
import unit_test as utest

commit_a = rarg.args.commit_a
commit_b = rarg.args.commit_b

root_path = io.add_trailing_slash('/home/fenics/shared')

print(f'commit_a = {commit_a}')
print(f'commit_b = {commit_b}')

problem = 'square_a'
code_path = root_path + 'steady-state-no-flow'
mesh_path = root_path + 'generate_mesh/2d/square'

utest.checkout(commit_a)
utest.go_to_path(mesh_path)
os.system(f'SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_mesh.py 0.1 $SOLUTION_PATH')

utest.go_to_path(code_path)
os.system(f'cd {code_path}; MESH_PATH="{mesh_path}/solution"; SOLUTION_PATH="{code_path}/solution"; rm -rf $SOLUTION_PATH; python3 solve.py {problem} $MESH_PATH $SOLUTION_PATH')

utest.checkout(commit_b)


utest.checkout('unit_test')


