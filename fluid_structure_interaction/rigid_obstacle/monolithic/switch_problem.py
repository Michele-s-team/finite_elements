import colorama as col

import runtime_arguments as rarg


if rarg.args.problem == 'square_polygon':

    fi = 'files_square_polygon'
    fsp = 'function_spaces_square_polygon'
    rmsh = 'mesh.read.square_polygon'
    vp = 'variational_problem_bc_square_polygon'
    prout_bc = 'print_out_bc_square_polygon'
    prout_ic = 'print_out_ic_square_polygon'
    prout_da = 'print_out_data_square_polygon'
    prout_sol = 'print_out_solution_square_polygon'



print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
