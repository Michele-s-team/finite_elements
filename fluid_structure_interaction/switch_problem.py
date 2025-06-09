import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_ellipse':
    rmsh = 'read_mesh_square_ellipse'
    vp = 'variational_problem_bc_square_ellipse'
    vp_pp = 'variational_problem_pp_square_ellipse'
    prout_bc = 'print_out_bc_square_ellipse'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square_ellipse'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
