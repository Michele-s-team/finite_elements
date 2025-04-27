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
    import read_mesh_square as rmsh
    import variational_problem_bc_square_a as vp
    import print_out_bc_square_a as prout_bc

elif rarg.args.problem == 'square_b':
    import read_mesh_square as rmsh
    import variational_problem_bc_square_b as vp
    import print_out_bc_square_b as prout_bc