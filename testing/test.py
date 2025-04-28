'''
this code performs a test of mlutiple parts of the code by comparing the solution in csv files across two commits

run with

python3 test.py [sha of commit_a] [sha of commit_b]
Example
    python3 test.py unit_test different_square
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import runtime_arguments as rarg
import command as cmd
import unit_test as utest

commit_a = rarg.args.commit_a
commit_b = rarg.args.commit_b

# the mesh resolution to be used to generated meshes across tests
mesh_resolution = 0.1

# the root path
root_path = io.add_trailing_slash('/home/fenics/shared')

print(f'commit_a = {commit_a}')
print(f'commit_b = {commit_b}')

# the paths where the mesh and problem solution will be stored, for both commits
mesh_solution_path_a = root_path + 'testing/commit_a/mesh/solution'
problem_solution_path_a = root_path + 'testing/commit_a/solution'
mesh_solution_path_b = root_path + 'testing/commit_b/mesh/solution'
problem_solution_path_b = root_path + 'testing/commit_b/solution'


# Compare commit_a and commit_b on a specific problem

# the path where the code which solves the variational problem is located
code_path = root_path + 'steady-state-no-flow'
# the path where the code which generates the mesh is locagted
mesh_path = root_path + 'generate_mesh/2d/square'
# name of the file which generates the mesh 
name_of_generate_mesh = 'generate_square_mesh'
# the variational problem to be solved from the code in code_path (it can be, for example, square_a, ring_1, ...)
problem = 'square_a'

#
# # checkout commit_a, generate the mesh and solve the problem
# cmd.checkout(commit_a)
#
# os.system(f'cd {mesh_path}; rm -rf {mesh_solution_path_a}; mkdir -p {mesh_solution_path_a}; python3 {name_of_generate_mesh}.py {mesh_resolution} {mesh_solution_path_a}')
# os.system(f'cd {code_path}; rm -rf {problem_solution_path_a}; mkdir -p {problem_solution_path_a}; python3 solve.py {problem} {mesh_solution_path_a} {problem_solution_path_a}')
#
# # checkout commit_b, generate the mesh and solve the problem
# cmd.checkout(commit_b)
# os.system(f'cd {mesh_path}; rm -rf {mesh_solution_path_b}; mkdir -p {mesh_solution_path_b}; python3 {name_of_generate_mesh}.py {mesh_resolution} {mesh_solution_path_b}')
# os.system(f'cd {code_path}; rm -rf {problem_solution_path_b}; mkdir -p {problem_solution_path_b}; python3 solve.py {problem} {mesh_solution_path_b} {problem_solution_path_b}')
#
# # compare the mesh and problem solution for commit_a and commit_b
# mesh_check = cmd.command_empty_err_out(f'cd {root_path}; ./compare-csv-files.sh {mesh_solution_path_a} {mesh_solution_path_b}')
# problem_check = cmd.command_empty_err_out(f'cd {root_path}; ./compare-csv-files.sh {problem_solution_path_a} {problem_solution_path_b}')
#
# # if check = true, then commit_a and commit_b give the same result
# check = (mesh_check and problem_check)
#
# io.check_print(mesh_check, f'Mesh check = {mesh_check}')
# io.check_print(problem_check, f'Problem check = {problem_check}')

utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path, mesh_path, code_path,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_mesh', mesh_resolution, 'square_b')

cmd.checkout('unit_test')
