import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_no_circle_a':
    rmsh = 'read_mesh_square_no_circle'
    vp = 'variational_problem_bc_square_no_circle_a'
    prout = 'print_out_bc_square_no_circle_a'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
