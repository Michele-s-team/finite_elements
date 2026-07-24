import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_no_circle':
    rmsh = 'mesh.read.square_no_circle'
    vp = 'variational_problem_bc_square_no_circle'
    vp_pp = 'variational_problem_pp_square_no_circle'
    prout_bc = 'print_out_bc_square_no_circle'
    prout_da = 'print_out_data_square_no_circle'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square_no_circle'



elif rarg.args.problem == 'square':
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square'
    vp_pp = 'variational_problem_pp_square'
    prout_bc = 'print_out_bc_square'
    prout_da = 'print_out_data_square'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
