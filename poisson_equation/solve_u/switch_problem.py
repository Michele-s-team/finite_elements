import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'ring_slice':

    rmsh = 'read_mesh_ring_slice'
    vp = 'variational_problem_bc_ring_slice'
    prout_bc = 'print_out_bc_ring_slice'

if rarg.args.problem == 'ring':

    rmsh = 'read_mesh_ring'
    vp = 'variational_problem_bc_ring'
    prout_bc = 'print_out_bc_ring'

if rarg.args.problem == 'ring_with_inner_circle':

    rmsh = 'read_mesh_ring_with_inner_circle'
    vp = 'variational_problem_bc_ring_with_inner_circle'
    prout_bc = 'print_out_bc_ring_with_inner_circle'

elif rarg.args.problem == 'square_no_circle':
    rmsh = 'read_mesh_square_no_circle'
    vp = 'variational_problem_bc_square_no_circle'
    prout_bc = 'print_out_bc_square_no_circle'

elif rarg.args.problem == 'two_squares_no_circle':
    rmsh = 'read_mesh_two_squares_no_circle'
    vp = 'variational_problem_bc_two_squares_no_circle'
    prout_bc =  'print_out_bc_two_squares_no_circle'

elif rarg.args.problem == 'square':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square'
    prout_bc =  'print_out_bc_square'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
