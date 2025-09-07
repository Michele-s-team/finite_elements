import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square':
    rmsh = 'mesh.read.square'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')
