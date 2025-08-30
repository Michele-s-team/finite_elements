import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_ellipse_circle':
    rmsh = 'mesh.read.square_ellipse_circle'
    vp_el = 'variational_problem_elastic_bc_square_ellipse_circle'
    vp_fl = 'variational_problem_fluid_bc_square_ellipse_circle'
    vp_msh = 'variational_problem_mesh_bc_square_ellipse_circle'
    vp_pp = 'variational_problem_pp_square_ellipse_circle'
    prout_bc = 'print_out_bc_square_ellipse_circle'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square_ellipse_circle'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
