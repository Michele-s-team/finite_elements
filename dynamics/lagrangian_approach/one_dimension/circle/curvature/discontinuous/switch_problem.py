import colorama as col

import runtime_arguments as rarg


if rarg.args.problem == 'square_shape_line_a':

    fi = 'files_square_shape_line_a'
    fsp = 'function_spaces_square_shape_line'
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_bc_square_shape_line_a'
    prout_bc = 'print_out_bc_square_shape_line_a'
    prout_ic = 'print_out_ic_square_shape_line'
    prout_da = 'print_out_data_square_shape_line'
    prout_sol = 'print_out_solution_square_shape_line'


elif rarg.args.problem == 'square_shape_line_b':

    fi = 'files_square_shape_line_b'
    fsp = 'function_spaces_square_shape_line'
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_bc_square_shape_line_b'
    prout_bc = 'print_out_bc_square_shape_line_b'
    prout_ic = 'print_out_ic_square_shape_line'
    prout_da = 'print_out_data_square_shape_line'
    prout_sol = 'print_out_solution_square_shape_line'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
