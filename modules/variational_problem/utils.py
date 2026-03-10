'''
this module contains methods used to handle variational problems
'''

from fenics import *


'''
set up and solve a variational problem
Input values: 
    * Mandatory: 
        - 'F': the variational functional
        - 'u': the function to solve for
        - 'J': the Jacobian
    * Optional:
        - 'params': a set of parameters for the solver, such as 
            params = {'nonlinear_solver': 'newton',
            'newton_solver':
                ...
            }
            'params' is None by default, and if it is !=None, the solver is initialized with parameters 'params'
'''

def solve_vp(F, u, bcs, J, params=None):

    J_u_sq = derivative(F, u, J)
    problem_u_sq = NonlinearVariationalProblem(F, u, bcs, J_u_sq)
    solver_u_sq = NonlinearVariationalSolver(problem_u_sq)

    if params != None:
        solver_u_sq.parameters.update(params)

    solver_u_sq.solve()