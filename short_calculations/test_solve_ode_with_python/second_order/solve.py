'''
solve the second-order boundary value problem

u'' - cos(u) = 0
u'(0) = 0, u(1) = 0

run with
    python3 solve.py

The solution can be compared with the Mathematica solution with the notebook check_the_solution.nb
'''

import colorama as col
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal


'''
set 
    y[0] = u
    y[0]' = y[1]
    y[1]' = cos(y[0])
    
This gives the following system of ODEs:

    y[i]' = F(y)[i], where
'''


def F(x, y):
    return np.vstack((y[1], np.cos(y[0])))


'''
this function enforces the boundary conditions: 
here ya is a vector containing [y[0], y[1]] at x = 0, and yb a vector containing [y[0], y[1]] at x=1
BCs returns a vector containing the residuals of the boundary conditions
So 'ya[1] - 1' means that y[1] = u' is enforced to be equal to 1 at x = 0, and 'yb[0] - 2' that u is enforced to be equal to 2 at x=
'''


def bcs(ya, yb):
    return np.array([ya[1] - 0, yb[0] - 0])


# number of bins in which the interval 0 <= x <= 1 is divided
N = 1024

x = np.linspace(0, 1, N)

y = np.zeros((2, x.size))

from scipy.integrate import solve_bvp

solve = solve_bvp(F, bcs, x, y)

x_output = np.linspace(0, 1, N)
y_output = solve.sol(x_output)[0]


print(f'Solution error = {col.Fore.RED}{ cal.error_solution_ode(0, 1, F, solve, N):.2e}{col.Fore.RESET}')
print(f'BCs error = {col.Fore.RED}{cal.error_bcs_ode(solve, 0, 1, bcs)}{col.Fore.RESET}')

# Create a DataFrame
df = pd.DataFrame({
    "x": x_output,
    "u(x)": y_output
})

# Save to CSV
df.to_csv("solution.csv", index=False)


plt.plot(x_output, y_output, label='y_a')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

