import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square'
    prout_bc =  'print_out_bc_square'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
