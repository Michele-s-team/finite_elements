import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_u_bc_square'
    vp_dot = 'variational_problem_u_dot_bc_square'
    prout_bc = 'print_out_u_bc_square'
    prout_bc_dot = 'print_out_u_dot_bc_square'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
