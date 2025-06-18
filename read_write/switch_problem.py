import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square':
    rmsh = 'read_mesh_square'

elif rarg.args.problem == 'square_ellipse':
    rmsh = 'read_mesh_square_ellipse'

elif rarg.args.problem == 'box_ball':
    rmsh = 'read_mesh_box_ball'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
