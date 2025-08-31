import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square':
    rmsh = 'mesh.read.square'

elif rarg.args.problem == 'square_ellipse':
    rmsh = 'mesh.read.square_ellipse'

elif rarg.args.problem == 'square_ellipse_circle':
    rmsh = 'mesh.read.square_ellipse_circle'

elif rarg.args.problem == 'box_ball':
    rmsh = 'mesh.read.box_ball'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
