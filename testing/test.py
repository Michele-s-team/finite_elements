'''
this code performs a test of mlutiple parts of the code by comparing the solution in csv files across two commits

run with

python3 test.py [sha of commit_a] [sha of commit_b]
Example
    clear; python3 test.py unit_test different_square
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
mesh_resolution = 0.1

root_path = io.add_trailing_slash('/home/fenics/shared')

print(f'commit_a = {commit_a}')
print(f'commit_b = {commit_b}')

problem = 'square_a'
code_path = root_path + 'steady-state-no-flow'
mesh_path = root_path + 'generate_mesh/2d/square'

mesh_solution_path_a = root_path + 'testing/commit_a/mesh/solution'
problem_solution_path_a = root_path + 'testing/commit_a/solution'

mesh_solution_path_b = root_path + 'testing/commit_b/mesh/solution'
problem_pp_tausolution_path_b = root_path + 'testing/commit_b/solution'



utest.checkout(commit_a)

os.system(f'cd {mesh_path}; rm -rf {mesh_solution_path_a}; mkdir -p {mesh_solution_path_a}; python3 generate_square_mesh.py {mesh_resolution} {mesh_solution_path_a}')
os.system(f'cd {code_path}; rm -rf {problem_solution_path_a}; mkdir -p {problem_solution_path_a}; python3 solve.py {problem} {mesh_solution_path_a} {problem_solution_path_a}')

utest.checkout(commit_b)
os.system(f'cd {mesh_path}; rm -rf {mesh_solution_path_b}; mkdir -p {mesh_solution_path_b}; python3 generate_square_mesh.py {mesh_resolution} {mesh_solution_path_b}')
os.system(f'cd {code_path}; rm -rf {problem_pp_tausolution_path_b}; mkdir -p {problem_pp_tausolution_path_b}; python3 solve.py {problem} {mesh_solution_path_b} {problem_pp_tausolution_path_b}')

#

output_out, output_err = utest.run_command(f'cd {root_path}; ./compare-csv-files.sh {mesh_solution_path_a} {mesh_solution_path_b}')
out_is_empty = (output_out.strip() == "")
err_is_empty = (output_err.strip() == "")
out_err_is_empty = (out_is_empty and err_is_empty)


print(f'Output = f{output_out.strip()}')
print(f'Error = f{output_err.strip()}')
print(f'Check ok = {col.Fore.YELLOW}{out_err_is_empty}{col.Fore.RESET}')

#

utest.checkout('unit_test')


