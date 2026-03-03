import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line':
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_line'
    prout_bc = 'print_out_bc_line'

elif rarg.args.problem == 'disk':
    rmsh = 'mesh.read.disk'
    vp = 'variational_problem_bc_disk'
    prout_bc = 'print_out_bc_disk'

elif rarg.args.problem == 'disk_robin':
    rmsh = 'mesh.read.disk'
    vp = 'variational_problem_bc_disk_robin'
    prout_bc = 'print_out_bc_disk_robin'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
