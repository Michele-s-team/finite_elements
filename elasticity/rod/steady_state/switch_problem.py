import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_no_circle_a':
    param = 'parameters_bc_square_no_circle_a'
    rmsh = 'read_mesh_square_no_circle'
    vp = 'variational_problem_u_bc_square_no_circle_a'
    vp_dot = 'variational_problem_u_dot_bc_square_no_circle_a'
    prout_bc = 'print_out_u_bc_square_no_circle_a'
    prout_bc_dot = 'print_out_u_dot_bc_square_no_circle_a'

elif rarg.args.problem == 'square_no_circle_b':
    param = 'parameters_bc_square_no_circle_b'
    rmsh = 'read_mesh_square_no_circle'
    vp = 'variational_problem_u_bc_square_no_circle_b'
    vp_dot = 'variational_problem_u_dot_bc_square_no_circle_b'
    prout_bc = 'print_out_u_bc_square_no_circle_b'
    prout_bc_dot = 'print_out_u_dot_bc_square_no_circle_b'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
