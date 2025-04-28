'''
this code performs a test of mlutiple parts of the code by comparing the solution in csv files across two commits

run with

python3 test.py [sha of commit_a] [sha of commit_b]
Example
    python3 test.py unit_test different
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

check = []

check.append(utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/ring',
                            root_path + 'steady-state-no-flow',
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_ring_mesh', mesh_resolution, 'ring'))

check.append(utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square_no_circle',
                            root_path + 'steady-state-no-flow',
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_no_circle_mesh', mesh_resolution, 'square_no_circle_a'))

check.append(utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square',
                            root_path + 'steady-state-no-flow',
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_mesh', mesh_resolution, 'square_a'))

check.append(utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square',
                            root_path + 'steady-state-no-flow',
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_mesh', mesh_resolution, 'square_b'))

cmd.checkout('unit_test')
