import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'ring':
    rmsh = 'mesh.read.ring'
    vp = 'variational_problem_bc_ring'
    prout_bc = 'print_out_bc_ring'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
