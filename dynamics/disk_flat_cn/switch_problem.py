import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'disk':
    rmsh = 'mesh.read.disk'
    vp = 'variational_problem_bc_disk'
    prout_bc = 'print_out_bc_disk'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
