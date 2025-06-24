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

# uncomment this after the merge with master - start
# problem_name = 'disk'
# generate_mesh_path = root_path + 'generate_mesh/2d/disk/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_disk_mesh', generate_mesh_path, 0.1, problem_name, success)
# uncomment this after the merge with master - end


problem_name = 'ring_slice'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/ring_slice/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh_ring_slice', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'half_circle_with_line'
generate_mesh_path = root_path + 'generate_mesh/2d/half_circle_with_line/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_half_circle_with_line_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'ring_symmetric'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/symmetric/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'ring_with_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/ring_with_circle/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_with_circle_mesh', generate_mesh_path, 0.1, problem_name, success)

# problem_name = 'square_no_circle'
# generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_square_no_circle_mesh', generate_mesh_path, 0.1, problem_name, success)
#
# problem_name = 'two_squares_no_circle'
# generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/two_squares_no_circle/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_two_squares_no_circle_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

# uncomment this after the merge with master - start
# problem_name = 'square'
# generate_mesh_path = root_path + 'generate_mesh/2d/square/lines/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_mesh', generate_mesh_path, 0.1, problem_name, success)
# uncomment this after the merge with master - end

#
# problem_name = 'square_symmetric_top_bottom'
# generate_mesh_path = root_path + 'generate_mesh/2d/square/symmetric_top_bottom/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square_symmetric_left_right_top_bottom'
generate_mesh_path = root_path + 'generate_mesh/2d/square/symmetric_left_right_top_bottom/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square_ellipse'
generate_mesh_path = root_path + 'generate_mesh/2d/square/ellipse/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_ellipse_mesh', generate_mesh_path, 0.1, problem_name, success)

#
# problem_name = 'ball'
# generate_mesh_path = root_path + 'generate_mesh/3d/ball/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_ball_mesh', 0.5, problem_name, success)
#
# problem_name = 'box'
# generate_mesh_path = root_path + 'generate_mesh/3d/box/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_box_mesh', 0.5, problem_name, success)
#
# problem_name = 'box_ball'
# generate_mesh_path = root_path + 'generate_mesh/3d/box_ball/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_box_ball_mesh', 0.5, problem_name, success)
#
# # Test poisson_equation/solve_u/periodic
# case_name = 'poisson_equation/solve_u/periodic'
#
# problem_name = 'square_no_circle'
# generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/symmetric/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_square_no_circle_mesh', 0.05, problem_name, success)
#
# # Test nitsche_method
#
#
# # Test nitsche_method/one_field
# case_name = 'nitsche_method/one_field'
#
# problem_name = 'square_no_circle'
# generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_square_no_circle_mesh', generate_mesh_path, 0.1, problem_name, success)

# Test nitsche_method/two_fields
case_name = 'nitsche_method/two_fields'

problem_name = 'disk'
generate_mesh_path = root_path + 'generate_mesh/2d/disk/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_disk_mesh', generate_mesh_path, 0.1, problem_name, success)


# Test fourth_order_pde
case_name = 'fourth_order_pde'

problem_name = 'ring_dirichlet'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'ring_nitsche'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

# Test fourth_order_pde/biharmonic_equation
case_name = 'fourth_order_pde/biharmonic_equation'

problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

# Test steady_state/no_flow
case_name = 'steady_state/no_flow'

problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/symmetric/'
checks[case_name + '_' + problem_name + '_symmetric'] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                                    root_path,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                                    'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)
#
# problem_name = 'square_no_circle_a'
# generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_square_no_circle_mesh', generate_mesh_path, 0.1, problem_name, success)
#
# problem_name = 'square_no_circle_a'
# generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/symmetric/'
# checks[case_name + '_' + problem_name + '_symmetric'] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                                     root_path,
#                                                                                     generate_mesh_path, root_path + case_name,
#                                                                                     generate_mesh_path, root_path + case_name,
#                                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                                     'generate_square_no_circle_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square_b'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

# Test steady_state/flow
case_name = 'steady_state/flow'

problem_name = 'ring_1'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'ring_1'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/symmetric/'
checks[case_name + '_' + problem_name + '_symmetric'] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                                    root_path,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                                    'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'ring_2'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'ring_2'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/symmetric/'
checks[case_name + '_' + problem_name + '_symmetric'] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                                    root_path,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                                    'generate_ring_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square_b'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path + 'high_res_parameters/', 0.01, problem_name, success)

# Test dynamics/channel_with_cylinder_flat_icps
case_name = 'dynamics/channel_with_cylinder_flat_icps'

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)
#
# problem_name = 'box_ball'
# generate_mesh_path =root_path + 'generate_mesh/3d/box_ball'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_box_ball_mesh', generate_mesh_path, 0.1, problem_name, success)
#

# Test dynamics/channel_with_cylinder_flat_cn
case_name = 'dynamics/channel_with_cylinder_flat_cn'

# problem_name = 'square_no_circle'
# generate_mesh_path =root_path + 'generate_mesh/2d/square_no_circle'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_square_no_circle_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

# Test dynamics/channel_with_cylinder_curved_cn
case_name = 'dynamics/channel_with_cylinder_curved_cn'

# problem_name = 'square_no_circle'
# generate_mesh_path =root_path + 'generate_mesh/2d/square_no_circle'
# checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
#                                                                      root_path,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      generate_mesh_path, root_path + case_name,
#                                                                      mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
#                                                                      'generate_square_no_circle_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

# Test dynamics
case_name = 'dynamics'

problem_name = 'square_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)

problem_name = 'square_b'
generate_mesh_path =root_path + 'generate_mesh/2d/square'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_mesh', generate_mesh_path, 0.1, problem_name, success)


# Test fluid_structure_interaction
case_name = 'fluid_structure_interaction/mesh_deformation'

problem_name = 'square_ellipse'
generate_mesh_path =root_path + 'generate_mesh/2d/square/ellipse'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_ellipse_mesh', generate_mesh_path, 0.1, problem_name, success)


case_name = 'fluid_structure_interaction'

problem_name = 'square_ellipse'
generate_mesh_path =root_path + 'generate_mesh/2d/square/ellipse'
checks[case_name + '_' + problem_name] = utest.test_problem_and_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_square_ellipse_mesh', generate_mesh_path, 0.1, problem_name, success)


cmd.checkout(commit_a, success)

max_key_len = max(len(key) for key in checks.keys())

for key, value in checks.items():
    status = io.check_string(value, "OK", "NOT OK")
    dots = '.' * (max_key_len + 10 - len(key))  # 5 is for minimum spacing
    print(f'{key} {dots} {status}')

total_test = all(list(checks.values()))

print(f'List of tests = {checks}')

io.print_star_box(f"Test = {total_test}", success=total_test)
