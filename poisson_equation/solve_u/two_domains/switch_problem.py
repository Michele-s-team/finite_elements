import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_square':
    rmsh = 'read_mesh_square_square'
    vp = 'variational_problem_bc_square_square'
    prout_bc = 'print_out_bc_square_square'

elif rarg.args.problem == 'square_ellipse_circle':
    rmsh = 'read_mesh_square_ellipse_circle'
    vp = 'variational_problem_bc_square_ellipse_circle'
    prout_bc = 'print_out_bc_square_ellipse_circle'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
