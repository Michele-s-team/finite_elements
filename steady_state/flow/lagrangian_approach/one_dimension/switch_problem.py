import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line_solve_nu':
    fsp = 'function_spaces_solve_nu'
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_line_solve_nu'
    prout_bc = 'print_out_bc_line_solve_nu'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
