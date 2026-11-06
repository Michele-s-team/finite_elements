import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'disk_a':
    rmsh = 'mesh.read.disk'
    vp = 'variational_problem_bc_disk_a'
    prout_bc = 'print_out_bc_disk_a'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
