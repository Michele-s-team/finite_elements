import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square':
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square'
    vp_pp = 'variational_problem_pp_square'
    prout_bc = 'print_out_bc_square'
    
print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
