import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'ring_constraint_u_v':
    fsp = 'function_spaces_constraint_u_v'
    rmsh = 'read_mesh_ring'
    vp = 'variational_problem_bc_ring_constraint_u_v'
    prout_bc = 'print_out_bc_ring_constraint_u_v'

elif rarg.args.problem == 'ring_constraint_grad_u_grad_v':
    fsp = 'function_spaces_constraint_grad_u_grad_v'
    rmsh = 'read_mesh_ring'
    vp = 'variational_problem_bc_ring_constraint_grad_u_grad_v'
    prout_bc = 'print_out_bc_ring_constraint_grad_u_grad_v'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
