import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_shape_line_a':
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_bc_square_shape_line_a'
    prout_bc = 'print_out_bc_square_shape_line_a'

elif rarg.args.problem == 'two_squares_no_circle_a':
    rmsh = 'mesh.read.two_squares_no_circle'
    vp = 'variational_problem_bc_two_squares_no_circle_a'
    prout_bc = 'print_out_bc_wo_squares_no_circle_a'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
