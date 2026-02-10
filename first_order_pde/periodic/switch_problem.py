import colorama as col

import runtime_arguments as rarg



if rarg.args.problem == 'line_scalar':

    fsp = 'function_spaces_bc_line_scalar'
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_line_scalar'
    prout_bc = 'print_out_bc_line_scalar'
    prout_sol = 'print_out_solution_scalar'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
