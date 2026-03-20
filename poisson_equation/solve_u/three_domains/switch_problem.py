import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_disk_line':
    rmsh = 'mesh.read.square_disk_line'
    vp_sub_mesh_0_0 = 'variational_problem_square_disk_line_sub_mesh_0_0'
    vp_sub_mesh_0_1 = 'variational_problem_square_disk_line_sub_mesh_0_1'
    vp_mesh_1 = 'variational_problem_square_disk_line_mesh_1'
    prout_bc = 'print_out_square_disk_line'

elif rarg.args.problem == 'square_shape_line':
    rmsh = 'mesh.read.square_shape_line'
    vp_sub_mesh_0_0 = 'variational_problem_square_shape_line_sub_mesh_0_0'
    vp_sub_mesh_0_1 = 'variational_problem_square_shape_line_sub_mesh_0_1'
    vp_mesh_1 = 'variational_problem_square_shape_line_mesh_1'
    prout_bc = 'print_out_square_shape_line'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
