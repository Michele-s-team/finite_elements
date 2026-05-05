import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_shape_line_a':

    fsp = 'function_spaces_two_meshes'
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_bc_square_shape_line_a'
    prout_sol = 'print_out_solution_square_shape_line'
    prout_bc = 'print_out_bc_square_shape_line_a'

elif rarg.args.problem == 'square_shape_line_b':

    fsp = 'function_spaces_two_meshes'
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_bc_square_shape_line_b'
    prout_sol = 'print_out_solution_square_shape_line'
    prout_bc = 'print_out_bc_square_shape_line_b'

elif rarg.args.problem == 'square_shape_line_c':

    fsp = 'function_spaces_two_meshes'
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_bc_square_shape_line_c'
    prout_sol = 'print_out_solution_square_shape_line'
    prout_bc = 'print_out_bc_square_shape_line_c'

elif rarg.args.problem == 'square_shape_line_d':

    fsp = 'function_spaces_two_meshes'
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_bc_square_shape_line_d'
    prout_sol = 'print_out_solution_square_shape_line'
    prout_bc = 'print_out_bc_square_shape_line_d'

elif rarg.args.problem == 'square_ellipse_circle_a':

    fsp = 'function_spaces_one_mesh'
    rmsh = 'mesh.read.square_ellipse_circle'
    vp = 'variational_problem_bc_square_ellipse_circle_a'
    prout_sol = 'print_out_solution_square_ellipse_circle'
    prout_bc = 'print_out_bc_square_ellipse_circle_a'

elif rarg.args.problem == 'two_squares_no_circle_a':

    fsp = 'function_spaces_one_mesh'
    rmsh = 'mesh.read.two_squares_no_circle'
    vp = 'variational_problem_bc_two_squares_no_circle_a'
    prout_sol = 'print_out_solution_two_squares_no_circle'
    prout_bc = 'print_out_bc_two_squares_no_circle_a'

elif rarg.args.problem == 'two_squares_no_circle_b':

    fsp = 'function_spaces_one_mesh'
    rmsh = 'mesh.read.two_squares_no_circle'
    vp = 'variational_problem_bc_two_squares_no_circle_b'
    prout_sol = 'print_out_solution_two_squares_no_circle'
    prout_bc = 'print_out_bc_two_squares_no_circle_b'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
