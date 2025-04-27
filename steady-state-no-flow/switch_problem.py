import runtime_arguments as rarg

if rarg.args.problem == 'ring':
    import read_mesh_ring as rmsh
    import variational_problem_bc_ring as vp
    import print_out_bc_ring as prout_bc

elif rarg.args.problem == 'square_no_circle_a':
    import read_mesh_square_no_circle as rmsh
    import variational_problem_bc_square_no_circle_a as vp
    import print_out_bc_square_no_circle_a as prout_bc

elif rarg.args.problem == 'square_a':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square_a'
    prout_bc =  'print_out_bc_square_a'

elif rarg.args.problem == 'square_b':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square_b'
    prout_bc =  'print_out_bc_square_b'