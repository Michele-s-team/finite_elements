import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line_a':
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_line_a'
    pr_bc =  'print_bcs_bc_line_a'
    pr_sol =  'print_solution'
    
elif rarg.args.problem == 'line_b':
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_line_b'
    pr_bc =  'print_bcs_bc_line_b'
    pr_sol =  'print_solution'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
