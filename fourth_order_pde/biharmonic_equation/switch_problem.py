import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line_vertex':
    rmsh = 'mesh.read.line_vertex'
    vp = 'variational_problem_bc_line_vertex'
    prout_bc = 'print_out_bc_line_vertex'

elif rarg.args.problem == 'ring':
    rmsh = 'mesh.read.ring'
    vp = 'variational_problem_bc_ring'
    prout_bc = 'print_out_bc_ring'



print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
