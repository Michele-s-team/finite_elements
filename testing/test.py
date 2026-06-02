'''
this code performs a test of mlutiple parts of the code by comparing the solution in csv files across two commits

run with

python3 test.py [sha of commit_a] [sha of commit_b]
Example
    python3 test.py master different
'''

import os
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
mesh_check_solution_path_a = root_path + 'testing/commit_a/mesh/check'
problem_solution_path_a = root_path + 'testing/commit_a/solution'
mesh_solution_path_b = root_path + 'testing/commit_b/mesh/solution'
mesh_check_solution_path_b = root_path + 'testing/commit_b/mesh/check'
problem_solution_path_b = root_path + 'testing/commit_b/solution'

# Compare commit_a and commit_b on a specific problem

checks = {}
success = [True]

cmd.run_command('clear; clear', success)

################################################################
# Mesh checks test
################################################################



'''
# 1d meshes

#line mesh
generate_mesh_path = root_path + 'generate_mesh/1d/line/'

checks[generate_mesh_path] = utest.test_generate_mesh_and_check_mesh(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, generate_mesh_path,
                                                                     generate_mesh_path, generate_mesh_path,
                                                                     mesh_solution_path_a, mesh_check_solution_path_a,
                                                                     mesh_solution_path_b, mesh_check_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', 'check_mesh',
                                                                     generate_mesh_path, generate_mesh_path,
                                                                     success)

'''

################################################################
# Variational problems test
################################################################

# Test first_order_pde/periodic
case_name = 'first_order_pde/periodic'

problem_name = 'line_scalar'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'line_vector'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test poisson_equation/solve_u
case_name = 'poisson_equation/solve_u'

problem_name = 'line'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'line_vertex'
generate_mesh_path = root_path + 'generate_mesh/1d/line/vertex/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)




problem_name = 'disk'
generate_mesh_path = root_path + 'generate_mesh/2d/disk/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'disk_vertices'
generate_mesh_path = root_path + 'generate_mesh/2d/disk/vertices'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'disk_vertices_tangent'
generate_mesh_path = root_path + 'generate_mesh/2d/disk/vertices'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


problem_name = 'ring_slice'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/ring_slice/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'half_circle_with_line'
generate_mesh_path = root_path + 'generate_mesh/2d/half_circle_with_line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'ring_symmetric'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/symmetric/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'ring_with_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/ring_with_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_no_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_no_circle_mirror'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'two_squares_no_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/two_squares_no_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/lines/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


problem_name = 'square_symmetric_top_bottom'
generate_mesh_path = root_path + 'generate_mesh/2d/square/symmetric_top_bottom/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_symmetric_left_right_top_bottom'
generate_mesh_path = root_path + 'generate_mesh/2d/square/symmetric_left_right_top_bottom/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_ellipse'
generate_mesh_path = root_path + 'generate_mesh/2d/square/ellipse/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_ellipse_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square/ellipse_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_polygon'
generate_mesh_path = root_path + 'generate_mesh/2d/square/polygon/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_half_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square/half_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


problem_name = 'ball'
generate_mesh_path = root_path + 'generate_mesh/3d/ball/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'box'
generate_mesh_path = root_path + 'generate_mesh/3d/box/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'box_ball'
generate_mesh_path = root_path + 'generate_mesh/3d/box_ball/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

# Test poisson_equation/solve_u/periodic
case_name = 'poisson_equation/solve_u/periodic'

problem_name = 'square_no_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/symmetric/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'line'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test poisson_equation/solve_u/two_domains
case_name = 'poisson_equation/solve_u/two_domains'


problem_name = 'square_square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


problem_name = 'square_ellipse_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square/ellipse_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



problem_name = 'square_no_circle_line'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test poisson_equation/solve_u/two_domains/discontinuous
case_name = 'poisson_equation/solve_u/two_domains/discontinuous'

problem_name = 'square_shape_line_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square/shape_line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_shape_line_b'
generate_mesh_path = root_path + 'generate_mesh/2d/square/shape_line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_shape_line_c'
generate_mesh_path = root_path + 'generate_mesh/2d/square/shape_line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_shape_line_d'
generate_mesh_path = root_path + 'generate_mesh/2d/square/shape_line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_ellipse_circle_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square/ellipse_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', 
                                                                    os.path.join(generate_mesh_path, 'poisson_equation_two_domains_discontinuous_parameters'), 
                                                                    os.path.join(generate_mesh_path, 'poisson_equation_two_domains_discontinuous_parameters'), 
                                                                    problem_name, problem_name, success)


problem_name = 'two_squares_no_circle_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/two_squares_no_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'two_squares_no_circle_b'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/two_squares_no_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



# Test poisson_equation/solve_u/three_domains
case_name = 'poisson_equation/solve_u/three_domains'

problem_name = 'square_shape_line'
generate_mesh_path = root_path + 'generate_mesh/2d/square/shape_line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  
                                                                     os.path.join(generate_mesh_path, 'disk_parameters'),  
                                                                     os.path.join(generate_mesh_path, 'disk_parameters'), 
                                                                     problem_name, problem_name, success)


# Test poisson_equation/solve_u/three_domains/transfer_test
case_name = 'poisson_equation/solve_u/three_domains/transfer_test'

problem_name = 'square_shape_line'
generate_mesh_path = root_path + 'generate_mesh/2d/square/shape_line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test poisson_equation/solve_u/discontinuous
case_name = 'poisson_equation/solve_u/discontinuous'

problem_name = 'disk_a'
generate_mesh_path = root_path + 'generate_mesh/2d/disk/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'disk_vertices_a'
generate_mesh_path = root_path + 'generate_mesh/2d/disk/vertices'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test poisson_equation/solve_u_v
case_name = 'poisson_equation/solve_u_v'

problem_name = 'line'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)




problem_name = 'disk'
generate_mesh_path = root_path + 'generate_mesh/2d/disk/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'disk_robin'
generate_mesh_path = root_path + 'generate_mesh/2d/disk/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)





# Test poisson_equation/constraint
case_name = 'poisson_equation/constraint'

problem_name = 'ring_constraint_u_v'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'ring_constraint_grad_u_grad_v'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



problem_name = 'ring_constraint_u2_v2'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



# Test nitsche_method


# Test nitsche_method/one_field
case_name = 'nitsche_method/one_field'


problem_name = 'line'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


problem_name = 'square_no_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)





# Test third_order_pde
case_name = 'third_order_pde'

problem_name = 'line_vertex_dirichlet'
generate_mesh_path = root_path + 'generate_mesh/1d/line/vertex'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  os.path.join(generate_mesh_path, 'resolution_0.1'), os.path.join(generate_mesh_path, 'resolution_0.1'), problem_name, problem_name, success)

problem_name = 'line_vertex_nitsche'
generate_mesh_path = root_path + 'generate_mesh/1d/line/vertex'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  os.path.join(generate_mesh_path, 'resolution_0.1'), os.path.join(generate_mesh_path, 'resolution_0.1'), problem_name, problem_name, success)


# Test fourth_order_pde
case_name = 'fourth_order_pde'

problem_name = 'line_vertex_dirichlet'
generate_mesh_path = root_path + 'generate_mesh/1d/line/vertex'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  os.path.join(generate_mesh_path, 'resolution_0.1'), os.path.join(generate_mesh_path, 'resolution_0.1'), problem_name, problem_name, success)

problem_name = 'line_vertex_nitsche'
generate_mesh_path = root_path + 'generate_mesh/1d/line/vertex'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  os.path.join(generate_mesh_path, 'resolution_0.1'), os.path.join(generate_mesh_path, 'resolution_0.1'), problem_name, problem_name, success)

problem_name = 'ring_dirichlet'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'ring_nitsche'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

# Test fourth_order_pde/biharmonic_equation
case_name = 'fourth_order_pde/biharmonic_equation'


problem_name = 'line_vertex'
generate_mesh_path = root_path + 'generate_mesh/1d/line/vertex'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  os.path.join(generate_mesh_path, 'resolution_0.1'), os.path.join(generate_mesh_path, 'resolution_0.1'), problem_name, problem_name, success)


problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test fourth_order_pde/constraint/u_v
case_name = 'fourth_order_pde/constraint/u_v'

problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



# Test fourth_order_pde/constraint/grad_u_grad_v
case_name = 'fourth_order_pde/constraint/grad_u_grad_v'

problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)








# Test steady_state/no_flow
case_name = 'steady_state/no_flow'

problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)




problem_name = 'ring'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/symmetric/'
checks[case_name + '_' + problem_name + '_symmetric'] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                                    root_path,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                                    'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_no_circle_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_no_circle_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/symmetric/'
checks[case_name + '_' + problem_name + '_symmetric'] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                                    root_path,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                                    'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_b'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



# Test steady_state/no_flow/lagrangian_approach/spherically_symmetric
case_name = 'steady_state/no_flow/lagrangian_approach/spherically_symmetric'

problem_name = 'square_no_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/symmetric/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path + 's_theta_parameters/', generate_mesh_path + 's_theta_parameters/', problem_name, problem_name, success)






# Test steady_state/no_flow/lagrangian_approach/one_dimension
case_name = 'steady_state/no_flow/lagrangian_approach/one_dimension'

problem_name = 'line_fixed_nu'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


problem_name = 'line_solve_nu'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)




# Test steady_state/flow
case_name = 'steady_state/flow'

problem_name = 'ring_1'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'ring_1'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/symmetric/'
checks[case_name + '_' + problem_name + '_symmetric'] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                                    root_path,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                                    'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'ring_2'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'ring_2'
generate_mesh_path = root_path + 'generate_mesh/2d/ring/symmetric/'
checks[case_name + '_' + problem_name + '_symmetric'] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                                    root_path,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    generate_mesh_path, root_path + case_name,
                                                                                    mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                                    'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_b'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path + 'high_res_parameters/', generate_mesh_path + 'high_res_parameters/', problem_name, problem_name, success)




# Test steady_state/flow/lagrangian_approach/one_dimension
case_name = 'steady_state/flow/lagrangian_approach/one_dimension'

problem_name = 'line_fixed_nu'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


problem_name = 'line_solve_nu'
generate_mesh_path = root_path + 'generate_mesh/1d/line/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



# Test dynamics/channel_with_cylinder_flat_icps
case_name = 'dynamics/channel_with_cylinder_flat_icps'

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'box_ball'
generate_mesh_path =root_path + 'generate_mesh/3d/box_ball'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test dynamics/channel_with_cylinder_flat_cn
case_name = 'dynamics/channel_with_cylinder_flat_cn'

problem_name = 'square_no_circle'
generate_mesh_path =root_path + 'generate_mesh/2d/square_no_circle'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


problem_name = 'square_half_circle'
generate_mesh_path = root_path + 'generate_mesh/2d/square/half_circle/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test dynamics/channel_with_cylinder_flat_cn/discontinuous
case_name = 'dynamics/channel_with_cylinder_flat_cn/discontinuous'

problem_name = 'square_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test dynamics/channel_with_cylinder_flat_cn/discontinuous/monolithic
case_name = 'dynamics/channel_with_cylinder_flat_cn/discontinuous/monolithic'

problem_name = 'square_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test dynamics/channel_with_cylinder_flat_cn
case_name = 'dynamics/channel_with_cylinder_flat_cn/monolithic'

problem_name = 'square'
generate_mesh_path =root_path + 'generate_mesh/2d/square'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



# Test dynamics/channel_with_cylinder_flat_cn/discontinuous/mixed_space
case_name = 'dynamics/channel_with_cylinder_flat_cn/discontinuous/mixed_space'

problem_name = 'square_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test dynamics/disk_flat_cn
case_name = 'dynamics/disk_flat_cn'

problem_name = 'disk'
generate_mesh_path =root_path + 'generate_mesh/2d/disk'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

# Test dynamics/channel_with_cylinder_curved_cn
case_name = 'dynamics/channel_with_cylinder_curved_cn'

problem_name = 'square_no_circle'
generate_mesh_path =root_path + 'generate_mesh/2d/square_no_circle'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square'
generate_mesh_path = root_path + 'generate_mesh/2d/square/'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

# Test dynamics
case_name = 'dynamics'

problem_name = 'square_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

problem_name = 'square_b'
generate_mesh_path =root_path + 'generate_mesh/2d/square'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test dynamics/lagrangian_approach/one_dimension
case_name = 'dynamics/lagrangian_approach/one_dimension/line'

problem_name = 'line_a'
generate_mesh_path =root_path + 'generate_mesh/1d/line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  
                                                                     os.path.join(generate_mesh_path, 'resolution_0.1'), os.path.join(generate_mesh_path, 'resolution_0.1'), 
                                                                     problem_name, problem_name, success)


problem_name = 'line_b'
generate_mesh_path =root_path + 'generate_mesh/1d/line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  
                                                                     os.path.join(generate_mesh_path, 'resolution_0.1'), os.path.join(generate_mesh_path, 'resolution_0.1'), 
                                                                     problem_name, problem_name, success)



case_name = 'dynamics/lagrangian_approach/one_dimension/circle'

problem_name = 'circle'
generate_mesh_path =root_path + 'generate_mesh/1d/line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  
                                                                     generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

case_name = 'dynamics/lagrangian_approach/one_dimension/circle/curvature'

problem_name = 'square_shape_line_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square/shape_line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',    
                                                                     os.path.join(generate_mesh_path, 'curvature_parameters'),  
                                                                     os.path.join(generate_mesh_path, 'curvature_parameters'), 
                                                                     problem_name, problem_name, success)


case_name = 'dynamics/lagrangian_approach/one_dimension/circle/curvature/discontinuous'

problem_name = 'square_shape_line_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square/shape_line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',    
                                                                     generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



# Test fluid_structure_interaction
case_name = 'fluid_structure_interaction/mesh_deformation'

problem_name = 'square_ellipse'
generate_mesh_path =root_path + 'generate_mesh/2d/square/ellipse'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


case_name = 'fluid_structure_interaction/rigid_obstacle'

problem_name = 'square_ellipse'
generate_mesh_path =root_path + 'generate_mesh/2d/square/ellipse'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


case_name = 'fluid_structure_interaction/rigid_obstacle/remesh'

problem_name = 'square_polygon'
generate_mesh_path =root_path + 'generate_mesh/2d/square/polygon'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',   generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)



case_name = 'fluid_structure_interaction/elastic_obstacle'

problem_name = 'square_ellipse_circle'
generate_mesh_path =root_path + 'generate_mesh/2d/square/ellipse_circle'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


case_name = 'fluid_structure_interaction/elastic_obstacle/monolithic'

problem_name = 'square_shape_line_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square/shape_line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  
                                                                     os.path.join(generate_mesh_path, 'elastic_obstacle_monolithic_square_shape_line_a_parameters'),
                                                                     os.path.join(generate_mesh_path, 'elastic_obstacle_monolithic_square_shape_line_a_parameters'), 
                                                                     problem_name, problem_name, success)

problem_name = 'square_shape_line_b'
generate_mesh_path =root_path + 'generate_mesh/2d/square/shape_line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

case_name = 'fluid_structure_interaction/membrane'

problem_name = 'square_no_circle_line_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square_no_circle/line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, 
                                                                     problem_name, problem_name, success)




case_name = 'fluid_structure_interaction/fluid_obstacle/remesh'

problem_name = 'square_shape_line_a'
generate_mesh_path =root_path + 'generate_mesh/2d/square/shape_line'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh', 
                                                                     generate_mesh_path, generate_mesh_path,         
                                                                     problem_name, problem_name, success)




# Test elasticity/rod/steady_state
case_name = 'elasticity/rod/steady_state'

problem_name = 'square_no_circle_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)

# Test elasticity/rod/steady_state/periodic
case_name = 'elasticity/rod/steady_state/periodic'

problem_name = 'square_no_circle_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle/symmetric'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)


# Test elasticity/rod/dynamics
case_name = 'elasticity/rod/dynamics'

problem_name = 'square_no_circle_a'
generate_mesh_path = root_path + 'generate_mesh/2d/square_no_circle'
checks[case_name + '_' + problem_name] = utest.test_generate_mesh_and_solve(commit_a, commit_b,
                                                                     root_path,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     generate_mesh_path, root_path + case_name,
                                                                     mesh_solution_path_a, problem_solution_path_a, mesh_solution_path_b, problem_solution_path_b,
                                                                     'generate_mesh', 'generate_mesh',  generate_mesh_path, generate_mesh_path, problem_name, problem_name, success)






cmd.checkout(commit_a, success)

max_key_len = max(len(key) for key in checks.keys())

for key, value in checks.items():
    status = io.check_string(value, "OK", "NOT OK")
    dots = '.' * (max_key_len + 10 - len(key))  # 5 is for minimum spacing
    print(f'{key} {dots} {status}')

total_test = all(list(checks.values()))

print(f'List of tests = {checks}')

io.print_star_box(f"Test = {total_test}", success=total_test)
