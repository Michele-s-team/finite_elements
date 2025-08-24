import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line':
    fsp = 'function_spaces_1d'
    rmsh = 'read_mesh_line'
    vp = 'variational_problem_bc_line'
    prout_bc = 'print_out_bc_line'

elif rarg.args.problem == 'disk':
    fsp = 'function_spaces_2d'
    rmsh = 'read_mesh_disk'
    vp = 'variational_problem_bc_disk'
    prout_bc = 'print_out_bc_disk'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
