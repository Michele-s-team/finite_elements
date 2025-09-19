import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line':
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_line'
    pr_bc =  'print_bcs_bc_line'
    pr_sol =  'print_solution_bc_line'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
