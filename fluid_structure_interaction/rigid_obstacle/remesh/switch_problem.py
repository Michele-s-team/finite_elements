import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_polygon':
    ap_shape = 'algebraic_problem_shape_bc_square_polygon'
    fi = 'files_bc_square_polygon'
    prout_bc = 'print_out_bc_square_polygon'
    prout_da = 'print_out_data_square_polygon'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square_polygon'
    rmsh = 'mesh.read.square_polygon'
    vp_fluid = 'variational_problem_fluid_bc_square_polygon'
    vp_mesh = 'variational_problem_mesh_bc_square_polygon'
    vp_pp = 'variational_problem_pp_square_polygon'
    
print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
