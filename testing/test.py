'''
this code performs a test of mlutiple parts of the code by comparing the solution in csv files across two commits

run with

python3 test.py [sha of commit_a] [sha of commit_b]
Example
    python3 test.py master different
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

checks = {}
success = [True]

cmd.run_command('clear; clear', success)


# Test poisson_equation/solve_u
case_name = 'poisson_equation/solve_u'

problem_name = 'ring_slice'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/ring/ring_slice',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh_ring_slice', 0.1, problem_name, success)


problem_name = 'square_no_circle'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/square_no_circle',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_no_circle_mesh', 0.1, problem_name, success)

'''

problem_name = 'two_squares_no_circle'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/square_no_circle/two_squares_no_circle',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_two_squares_no_circle_mesh', 0.1, problem_name)

problem_name = 'square'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/square',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', 0.1, problem_name)



# Test poisson_equation/solve_u/periodic
case_name = 'poisson_equation/solve_u/periodic'

problem_name = 'square_no_circle'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/square_no_circle/symmetric',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_no_circle_mesh', 0.05, problem_name)



# Test steady-state-no-flow
case_name = 'steady-state-no-flow'

problem_name = 'ring'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/ring',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', 0.1, problem_name)

problem_name = 'square_no_circle_a'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/square_no_circle',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_no_circle_mesh', 0.1, problem_name)
problem_name = 'square_a'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/square',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', 0.1, problem_name)
problem_name = 'square_b'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     root_path + 'generate_mesh/2d/square',
                                                                     root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', 0.1, problem_name)

# Test steady-state-flow
case_name = 'steady-state-flow'

problem_name = 'ring_1'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/ring',
                            root_path + case_name,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_ring_mesh', 0.1, problem_name)

problem_name = 'ring_2'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/ring',
                            root_path + case_name,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_ring_mesh', 0.1, problem_name)

problem_name = 'square_a'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square',
                            root_path + case_name,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_mesh', 0.1, problem_name)

problem_name = 'square_b'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square',
                            root_path + case_name,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_mesh', 0.01, problem_name)


# Test channel-with-cylinder-curved-crank_nicholson-discretizations
case_name = 'channel-with-cylinder-curved-crank_nicholson-discretization'

problem_name = 'square_no_circle'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square_no_circle',
                            root_path + case_name,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_no_circle_mesh', 0.1, problem_name)

problem_name = 'square'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square',
                            root_path + case_name,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_mesh', 0.1, problem_name)


# Test dynamics
case_name = 'dynamics'

problem_name = 'square_a'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square',
                            root_path + case_name,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_mesh', 0.1, problem_name)

problem_name = 'square_b'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                            root_path,
                            root_path + 'generate_mesh/2d/square',
                            root_path + case_name,
                            mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                            'generate_square_mesh', 0.1, problem_name)
'''

cmd.checkout(commit_a, success)



max_key_len = max(len(key) for key in checks.keys())

for key, value in checks.items():
    status = io.check_string(value, "OK", "NOT OK")
    dots = '.' * (max_key_len + 10 - len(key))  # 5 is for minimum spacing
    print(f'{key} {dots} {status}')

total_test = all(list(checks.values()))

print(f'List of tests = {checks}')

io.print_star_box(f"Test = {total_test}", success=total_test)
