import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_ellipse_circle':
    rmsh = 'mesh.read.square_ellipse_circle'
    vp = 'variational_problem_bc_square_ellipse_circle'
    prout_bc = 'print_out_bc_square_ellipse_circle'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square_ellipse_circle'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
