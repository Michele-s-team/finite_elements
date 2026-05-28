'''
solve the fourth-order boundary value problem for steady_state_no_flow

run with
    python3 solve.py

The solution can be compared with the Mathematica solution with the notebook check_the_solution.nb
'''

import colorama as col
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_bvp
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal

def sigma(x):
    return 1


kappa = 1.0
r = 1
R = 2

z_r = 0
z_R = 0.1
zp_r = 0.1
zp_R = 0.2

'''
set 
    y[0] = u
    y[0]' = y[1]
    y[1]' = y[2]
    y[2]' = y[3]
    y[3]' =  f(...)
    
This gives the following system of ODEs:

    y[i]' = F(y)[i], where
'''


def g(x, y):
    return -((2 * y[1] + 7 * y[1] ** 3 + 9 * y[1] ** 5 + 5 * y[1] ** 7 + y[1] ** 9 - 2 * x * y[2] - 3 * x * y[1] ** 2 * y[2] + x * y[1] ** 6 * y[2] - 15 * x ** 2 * y[1] * y[2] ** 2 - 15 * x ** 2 * y[1] ** 3 * y[2] ** 2 - 5 * x ** 3 * y[2] ** 3 + 30 * x ** 3 * y[1] ** 2 * y[2] ** 3 + 4 * x ** 2 * (1 + y[1] ** 2) * (1 + y[1] ** 2 - 5 * x * y[1] * y[2]) * y[3]) / (2 * x ** 3 * (1 + y[1] ** 2) ** 2)) + ((1 + y[1] ** 2) * (y[1] + y[1] ** 3 + x * y[2]) * sigma(x)) / (x * kappa)


def F(x, y):
    return np.vstack((y[1], y[2], y[3], g(x, y)))


'''
this function enforces the boundary conditions: 
here ya is a vector containing [y[0], y[1]] at x = 0, and yb a vector containing [y[0], y[1]] at x=1
BCs returns a vector containing the residuals of the boundary conditions
So 'ya[1] - 1' means that y[1] = u' is enforced to be equal to 1 at x = 0, and 'yb[0] - 2' that u is enforced to be equal to 2 at x=
'''



def bcs(ya, yb):
    return np.array([ya[0] - z_r, yb[0] - z_R, ya[1] - zp_r, yb[1] - zp_R])


# number of bins in which the interval 0 <= x <= 1 is divided
N = int(2e3)

x = np.linspace(r, R, N)

y = np.zeros((4, x.size))


solve = solve_bvp(F, bcs, x, y)

x_output = np.linspace(r, R, N)
y0_output = solve.sol(x_output)[0]
y1_output = solve.sol(x_output)[1]
y2_output = solve.sol(x_output)[2]
y3_output = solve.sol(x_output)[3]

print(f'Solution error = {col.Fore.RED}{ cal.error_solution_ode(r, R, F, solve, N):.2e}{col.Fore.RESET}')
print(f'BCs error = {col.Fore.RED}{cal.error_bcs_ode(solve, r, R, bcs)}{col.Fore.RESET}')


# Create a DataFrame
df = pd.DataFrame({
    "r": x_output,
    "z": y0_output,
    "z'": y1_output,
    "z''": y2_output,
    "z'''": y3_output
})

# Save to CSV
df.to_csv("solution.csv", index=False)

plt.plot(x_output, y0_output, label='z')
plt.plot(x_output, y1_output, label='z\'')
plt.plot(x_output, y2_output, label='z\'\'')
plt.plot(x_output, y3_output, label='z\'\'\'')
plt.legend()
plt.xlabel("r")
plt.ylabel("")
plt.show()
