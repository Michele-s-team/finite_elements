import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'line_vertex_dirichlet':
    rmsh = 'mesh.read.line_vertex'
    vp = 'variational_problem_bc_line_vertex_dirichlet'
    prout_bc = 'print_out_bc_line_vertex'
    
elif rarg.args.problem == 'line_vertex_nitsche':
    rmsh = 'mesh.read.line_vertex'
    vp = 'variational_problem_bc_line_vertex_nitsche'
    prout_bc = 'print_out_bc_line_vertex'

elif rarg.args.problem == 'ring_dirichlet':
    rmsh = 'mesh.read.ring'
    vp = 'variational_problem_bc_ring_dirichlet'
    prout_bc = 'print_out_bc_ring_dirichlet'

elif rarg.args.problem == 'ring_nitsche':
    rmsh = 'mesh.read.ring'
    vp = 'variational_problem_bc_ring_nitsche'
    prout_bc = 'print_out_bc_ring_nitsche'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
