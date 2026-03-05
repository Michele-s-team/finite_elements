import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_disk_line_a':
    rmsh = 'mesh.read.square_disk_line'
    vp_I = 'variational_problem_interface_square_disk_line_a'
    vp_D = 'variational_problem_domain_square_disk_line_a'
    vp_fluid_di = 'variational_problem_fluid_disk_square_disk_line_a'
    vp_fluid_sq = 'variational_problem_fluid_square_square_disk_line_a'
    vp_M = 'variational_problem_micelles_square_disk_line_a'
    prout_bc = 'print_out_square_disk_line_a'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
