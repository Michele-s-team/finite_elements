import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_a':
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square_a'
    prout_bc =  'print_out_bc_square_a'

elif rarg.args.problem == 'square_b':
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square_b'
    prout_bc =  'print_out_bc_square_b'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
