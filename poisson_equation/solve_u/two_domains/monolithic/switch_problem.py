import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_shape_line':
    rmsh = 'mesh.read.square_shape_line'
    vp = 'variational_problem_square_shape_line'
    prout_bc = 'print_out_square_shape_line'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
