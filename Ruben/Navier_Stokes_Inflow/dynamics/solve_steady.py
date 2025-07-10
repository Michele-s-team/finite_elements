#!/usr/bin/env python3
import sys, importlib
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver, parameters, DirichletBC, Constant
import colorama as col

import runtime_arguments as rarg
import switch_problem   as swi
import function_spaces_steady  as fsp
import print_out_solution_steady as pr_sol  
#import stability_operators as ops
import numpy as np

def solve_steady():
    # Load mesh and variational problem
    rmsh = importlib.import_module(swi.rmsh)
    vp   = importlib.import_module(f"variational_problem_bc_{rarg.args.problem}_steady")

    # Pull mixed-space and problem data
    W    = vp.W
    up   = vp.up
    bcs  = vp.bcs
    F    = vp.F
    J    = vp.J

    # Setup and solve nonlinear variational problem
    problem = NonlinearVariationalProblem(F, up, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["absolute_tolerance"]  = 1e-8
    solver.parameters["newton_solver"]["relative_tolerance"]  = 1e-6
    solver.parameters["newton_solver"]["maximum_iterations"]  = 50
    solver.solve()

    # Extract velocity and pressure
    u_star, p_star = up.split()

    # Output solutions
    pr_sol.print_solution_steady(u_star, p_star)
    print('Solved Steady-State Problem')

    # Return mesh module, velocity-space, velocity solution, and BCs
    return rmsh, fsp.Q_v, u_star, p_star