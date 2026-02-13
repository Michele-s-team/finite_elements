import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'circle':

    fsp = 'function_spaces'
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_circle'
    prout_bc = 'print_out_bc_circle'
    prout_sol = 'print_out_solution'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
