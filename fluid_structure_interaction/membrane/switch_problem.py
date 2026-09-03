import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_no_circle_line_a':
    fi = 'files_square_no_circle_line_a'
    rmsh = 'mesh.read.square_no_circle_line'
    vp_membrane = 'variational_problem_membrane_bc_square_no_circle_line_a'
    vp_mesh = 'variational_problem_mesh_bc_square_no_circle_line_a'
    vp_fluid = 'variational_problem_fluid_bc_square_no_circle_line_a'
    vp_pp = 'variational_problem_pp_square_no_circle_line_a'
    prout_bc = 'print_out_bc_square_no_circle_line_a'
    prout_da = 'print_out_data_square_no_circle_line_a'
    prout_sol = 'print_out_solution_square_no_circle_line_a'
    sh = 'curve_square_shape_line'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
