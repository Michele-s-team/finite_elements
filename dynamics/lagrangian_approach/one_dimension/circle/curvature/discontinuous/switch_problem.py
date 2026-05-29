import colorama as col

import runtime_arguments as rarg


if rarg.args.problem == 'square_shape_line_a':

    fsp = 'function_spaces_square_shape_line'
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_bc_square_shape_line_a'
    prout_sol = 'print_out_solution_square_shape_line_a'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
