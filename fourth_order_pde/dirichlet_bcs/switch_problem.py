import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'ring_dirichlet':
    rmsh = 'read_mesh_ring'
    vp = 'variational_problem_bc_ring_dirichlet'
    prout_bc = 'print_out_bc_ring_dirichlet'

elif rarg.args.problem == 'ring_nitsche':
    rmsh = 'read_mesh_ring'
    vp = 'variational_problem_bc_ring_nitsche'
    prout_bc = 'print_out_bc_ring_nitsche'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
