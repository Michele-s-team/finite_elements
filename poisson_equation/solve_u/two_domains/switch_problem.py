import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_square':
    rmsh = 'mesh.read.square_square'
    vp_sub_mesh_0 = 'variational_problem_square_square_sub_mesh_0'
    vp_sub_mesh_1 = 'variational_problem_square_square_sub_mesh_1'
    prout_bc = 'print_out_square_square'

elif rarg.args.problem == 'square_ellipse_circle':
    rmsh = 'mesh.read.square_ellipse_circle'
    vp_sub_mesh_0 = 'variational_problem_square_ellipse_circle_sub_mesh_0'
    vp_sub_mesh_1 = 'variational_problem_square_ellipse_circle_sub_mesh_1'
    prout_bc = 'print_out_square_ellipse_circle'

elif rarg.args.problem == 'square_no_circle_line':
    rmsh = 'mesh.read.square_no_circle_line'
    vp_sub_mesh_0 = 'variational_problem_square_no_circle_line_sub_mesh_0'
    vp_sub_mesh_1 = 'variational_problem_square_no_circle_line_sub_mesh_1'
    prout_bc = 'print_out_square_no_circle_line'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
