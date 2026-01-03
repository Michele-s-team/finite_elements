import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_no_circle':

    fsp = 'function_spaces_bc_square_no_circle'
    rmsh = 'mesh.read.square_no_circle'
    vp = 'variational_problem_bc_square_no_circle'
    prout_bc = 'print_out_bc_square_no_circle'

elif rarg.args.problem == 'line':

    fsp = 'function_spaces_bc_line'
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_line'
    prout_bc = 'print_out_bc_line'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
