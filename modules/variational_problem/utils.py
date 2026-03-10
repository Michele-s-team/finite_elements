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
        - 'parameters': a set of parameters for the solver, such as 
            parameters = {'nonlinear_solver': 'newton',
            'newton_solver':
                ...
            }
            'parameters' is None by default, and if it is !=None, the solver is initialized with parameters 'parameters'
'''

def solve_vp(F, u, bcs, J, parameters=None):

    J_der = derivative(F, u, J)
    variational_problem = NonlinearVariationalProblem(F, u, bcs, J_der)
    solver = NonlinearVariationalSolver(variational_problem)

    if parameters != None:
        solver.parameters.update(parameters)

    solver.solve()