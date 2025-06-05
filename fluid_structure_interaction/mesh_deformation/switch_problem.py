import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_ellipse':
    rmsh = 'read_mesh_square_ellipse'
    vp = 'variational_problem_u_bc_square_ellipse'
    vp_dot = 'variational_problem_u_dot_bc_square_ellipse'
    prout_bc = 'print_out_u_bc_square_ellipse'
    prout_bc_dot = 'print_out_u_dot_bc_square_ellipse'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
