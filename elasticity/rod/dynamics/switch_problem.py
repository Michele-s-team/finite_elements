import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_no_circle':
    rmsh = 'read_mesh_square_no_circle'
    vp = 'variational_problem_bc_square_no_circle'
    vp_dot = 'variational_problem_u_dot_bc_square_no_circle'
    prout_bc = 'print_out_bc_square_no_circle'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
