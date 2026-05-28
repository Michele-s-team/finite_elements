import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line':
    rmsh = 'mesh.read.line'
    vp = 'variational_problem_bc_line'
    prout_bc = 'print_out_bc_line'
    
elif rarg.args.problem == 'line_vertex':
    rmsh = 'mesh.read.line_vertex'
    vp = 'variational_problem_bc_line_vertex'
    prout_bc = 'print_out_bc_line_vertex'

elif rarg.args.problem == 'disk':
    rmsh = 'mesh.read.disk'
    vp = 'variational_problem_bc_disk'
    prout_bc = 'print_out_bc_disk'

elif rarg.args.problem == 'disk_vertices':
    rmsh = 'mesh.read.disk_vertices'
    vp = 'variational_problem_bc_disk_vertices'
    prout_bc = 'print_out_bc_disk_vertices'

elif rarg.args.problem == 'disk_vertices_tangent':
    rmsh = 'mesh.read.disk_vertices'
    vp = 'variational_problem_bc_disk_vertices_tangent'
    prout_bc = 'print_out_bc_disk_vertices_tangent'

elif rarg.args.problem == 'ring_slice':
    rmsh = 'mesh.read.ring_slice'
    vp = 'variational_problem_bc_ring_slice'
    prout_bc = 'print_out_bc_ring_slice'

elif rarg.args.problem == 'half_circle_with_line':

    rmsh = 'mesh.read.half_circle_with_line'
    vp = 'variational_problem_bc_half_circle_with_line'
    prout_bc = 'print_out_bc_half_circle_with_line'

elif rarg.args.problem == 'ring':

    rmsh = 'mesh.read.ring'
    vp = 'variational_problem_bc_ring'
    prout_bc = 'print_out_bc_ring'

elif rarg.args.problem == 'ring_symmetric':

    rmsh = 'mesh.read.ring'
    vp = 'variational_problem_bc_ring'
    prout_bc = 'print_out_bc_ring'

elif rarg.args.problem == 'ring_with_circle':

    rmsh = 'mesh.read.ring_with_circle'
    vp = 'variational_problem_bc_ring_with_circle'
    prout_bc = 'print_out_bc_ring_with_circle'

elif rarg.args.problem == 'square_no_circle':
    rmsh = 'mesh.read.square_no_circle'
    vp = 'variational_problem_bc_square_no_circle'
    prout_bc = 'print_out_bc_square_no_circle'
    
elif rarg.args.problem == 'square_no_circle_mirror':
    rmsh = 'mesh.read.square_no_circle'
    vp = 'variational_problem_bc_square_no_circle_mirror'
    prout_bc = 'print_out_bc_square_no_circle_mirror'

elif rarg.args.problem == 'two_squares_no_circle':
    rmsh = 'mesh.read.two_squares_no_circle'
    vp = 'variational_problem_bc_two_squares_no_circle'
    prout_bc = 'print_out_bc_two_squares_no_circle'

elif rarg.args.problem == 'square':
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square'
    prout_bc = 'print_out_bc_square'

elif rarg.args.problem == 'square_symmetric_top_bottom':
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square_symmetric_top_bottom'
    prout_bc = 'print_out_bc_square_symmetric_top_bottom'

elif rarg.args.problem == 'square_symmetric_left_right_top_bottom':
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square_symmetric_left_right_top_bottom'
    prout_bc = 'print_out_bc_square_symmetric_left_right_top_bottom'
    
elif rarg.args.problem == 'square_half_circle':
    rmsh = 'mesh.read.square_half_circle'
    vp = 'variational_problem_bc_square_half_circle'
    prout_bc = 'print_out_bc_square_half_circle'

elif rarg.args.problem == 'square_ellipse':
    rmsh = 'mesh.read.square_ellipse'
    vp = 'variational_problem_bc_square_ellipse'
    prout_bc = 'print_out_bc_square_ellipse'

elif rarg.args.problem == 'square_ellipse_circle':
    rmsh = 'mesh.read.square_ellipse_circle'
    vp = 'variational_problem_bc_square_ellipse_circle'
    prout_bc = 'print_out_bc_square_ellipse_circle'

elif rarg.args.problem == 'square_polygon':
    rmsh = 'mesh.read.square_polygon'
    vp = 'variational_problem_bc_square_polygon'
    prout_bc = 'print_out_bc_square_polygon'

elif rarg.args.problem == 'ball':
    rmsh = 'mesh.read.ball'
    vp = 'variational_problem_bc_ball'
    prout_bc = 'print_out_bc_ball'

elif rarg.args.problem == 'box':
    rmsh = 'mesh.read.box'
    vp = 'variational_problem_bc_box'
    prout_bc = 'print_out_bc_box'

elif rarg.args.problem == 'box_ball':
    rmsh = 'mesh.read.box_ball'
    vp = 'variational_problem_bc_box_ball'
    prout_bc = 'print_out_bc_box_ball'


print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
