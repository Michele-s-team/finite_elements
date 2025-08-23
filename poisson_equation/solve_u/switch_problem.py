import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line':
    rmsh = 'read_mesh_line'
    vp = 'variational_problem_bc_line'
    prout_bc = 'print_out_bc_line'

elif rarg.args.problem == 'disk':
    rmsh = 'read_mesh_disk'
    vp = 'variational_problem_bc_disk'
    prout_bc = 'print_out_bc_disk'

elif rarg.args.problem == 'ring_slice':
    rmsh = 'read_mesh_ring_slice'
    vp = 'variational_problem_bc_ring_slice'
    prout_bc = 'print_out_bc_ring_slice'

elif rarg.args.problem == 'half_circle_with_line':

    rmsh = 'read_mesh_half_circle_with_line'
    vp = 'variational_problem_bc_half_circle_with_line'
    prout_bc = 'print_out_bc_half_circle_with_line'

elif rarg.args.problem == 'ring':

    rmsh = 'read_mesh_ring'
    vp = 'variational_problem_bc_ring'
    prout_bc = 'print_out_bc_ring'

elif rarg.args.problem == 'ring_symmetric':

    rmsh = 'read_mesh_ring'
    vp = 'variational_problem_bc_ring'
    prout_bc = 'print_out_bc_ring'

elif rarg.args.problem == 'ring_with_circle':

    rmsh = 'read_mesh_ring_with_circle'
    vp = 'variational_problem_bc_ring_with_circle'
    prout_bc = 'print_out_bc_ring_with_circle'

elif rarg.args.problem == 'square_no_circle':
    rmsh = 'read_mesh_square_no_circle'
    vp = 'variational_problem_bc_square_no_circle'
    prout_bc = 'print_out_bc_square_no_circle'

elif rarg.args.problem == 'two_squares_no_circle':
    rmsh = 'read_mesh_two_squares_no_circle'
    vp = 'variational_problem_bc_two_squares_no_circle'
    prout_bc = 'print_out_bc_two_squares_no_circle'

elif rarg.args.problem == 'square':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square'
    prout_bc = 'print_out_bc_square'

elif rarg.args.problem == 'square_symmetric_top_bottom':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square_symmetric_top_bottom'
    prout_bc = 'print_out_bc_square_symmetric_top_bottom'

elif rarg.args.problem == 'square_symmetric_left_right_top_bottom':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square_symmetric_left_right_top_bottom'
    prout_bc = 'print_out_bc_square_symmetric_left_right_top_bottom'

elif rarg.args.problem == 'square_ellipse':
    rmsh = 'read_mesh_square_ellipse'
    vp = 'variational_problem_bc_square_ellipse'
    prout_bc = 'print_out_bc_square_ellipse'

elif rarg.args.problem == 'ball':
    rmsh = 'read_mesh_ball'
    vp = 'variational_problem_bc_ball'
    prout_bc = 'print_out_bc_ball'

elif rarg.args.problem == 'box':
    rmsh = 'read_mesh_box'
    vp = 'variational_problem_bc_box'
    prout_bc = 'print_out_bc_box'

elif rarg.args.problem == 'box_ball':
    rmsh = 'read_mesh_box_ball'
    vp = 'variational_problem_bc_box_ball'
    prout_bc = 'print_out_bc_box_ball'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
