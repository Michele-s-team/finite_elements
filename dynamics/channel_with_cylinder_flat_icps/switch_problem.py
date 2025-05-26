import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square'
    vp_pp = 'variational_problem_pp_square'
    prout_bc = 'print_out_bc_square'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square'

elif rarg.args.problem == 'box_ball':
    rmsh = 'read_mesh_box_ball'
    vp = 'variational_problem_bc_box_ball'
    vp_pp = 'variational_problem_pp_box_ball'
    prout_bc = 'print_out_bc_box_ball'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_box_ball'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
