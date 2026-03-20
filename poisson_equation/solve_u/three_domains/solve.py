'''
This code solves the Poisson equation in two sub_meshes, sub_mesh[0] and sub_mesh[1], which share one boundary
The problem is first solved in sub_mesh[1], and the solution u[1] is then used to specify the BCs of the problem of sub_mesh[0]

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]

Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/disk_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/three_domains/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_disk_line $MESH_PATH $SOLUTION_PATH
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/three_domains/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line $MESH_PATH $SOLUTION_PATH
 '''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp_mesh_0 = ['','']

# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }




'''
####################
# test transfer function

import numpy as np

delta_theta = 2 * np.pi / rmsh.lmsh.mesh_parameters[0]['N']
alpha = (np.pi - delta_theta)/2.0
delta_l = rmsh.lmsh.mesh_parameters[0]['r'] * 2.0 * np.sin(delta_theta/2.0)


# 1 transfer scalar
# 1.1 transfer from 2d to line 

class f_0_1_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)
    
fsp.f_sub_mesh_0_1.interpolate(f_0_1_Expression(element=fsp.Q[0][1].ufl_element()))


msh.transfer_circle_to_line(fsp.f_sub_mesh_0_1, fsp.f_mesh_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])

io.full_print(fsp.f_sub_mesh_0_1, f'u_2d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

io.full_print(fsp.f_mesh_1, f'u_line', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
    
print(f'Comparing the two functions on polygon vertices: ')
error = 0

for i in range(rmsh.lmsh.mesh_parameters[0]['N']):
    print(f'u_line = {fsp.f_mesh_1(i*delta_l)}\t u_2d = {fsp.f_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))}')

    a = fsp.f_mesh_1(i*delta_l)
    b = fsp.f_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))

    if abs(a-b) > error:
            error = abs(a-b)

print(f'error = {error}')



# 1.2 transfer from line  to 2d

# here one needs to choose a periodic analytical expression, because f_mesh_1 is defined on a periodic space Q[1]
class f_1_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = np.cos(2.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))

    def value_shape(self):
        return (1,)
    
fsp.f_mesh_1.interpolate(f_1_Expression(element=fsp.Q[1].ufl_element()))


msh.transfer_line_to_circle(fsp.f_mesh_1, fsp.f_sub_mesh_0_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])

io.full_print(fsp.f_sub_mesh_0_1, f'u_2d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

io.full_print(fsp.f_mesh_1, f'u_line', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
    
print(f'Comparing the two functions on polygon vertices: ')
error = 0

for i in range(rmsh.lmsh.mesh_parameters[0]['N']):
    print(f'u_line = {fsp.f_mesh_1(i*delta_l)}\t u_2d = {fsp.f_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))}')

    a = fsp.f_mesh_1(i*delta_l)
    b = fsp.f_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))

    if abs(a-b) > error:
            error = abs(a-b)

print(f'error = {error}')


# 2 transfer vector

# 2.1 transfer from 2d mesh to line mesh 
class v_sub_mesh_0_1_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 - x[0] + 2 * x[1] ** 2
        values[1] = 1 - 4 * x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (2,)
    
fsp.v_sub_mesh_0_1.interpolate(v_sub_mesh_0_1_Expression(element=fsp.V_sub_mesh_0_1.ufl_element()))


msh.transfer_circle_to_line(fsp.v_sub_mesh_0_1, fsp.v_mesh_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])

io.full_print(fsp.v_sub_mesh_0_1, f'v_2d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

io.full_print(fsp.v_mesh_1, f'v_line', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
    
print(f'Comparing the two functions on polygon vertices: ')
error = 0

for i in range(rmsh.lmsh.mesh_parameters[0]['N']):
    print(f'v_line = {fsp.v_mesh_1(i*delta_l)}\t v_2d = {fsp.v_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))}')

    a = fsp.v_mesh_1(i*delta_l)
    b = fsp.v_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))

    for j in range(len(a)):
        if abs(a[j]-b[j]) > error:
            error = abs(a[j]-b[j])

print(f'error = {error}')


# 2.2 transfer from line mesh to 2d mesh 
class v_mesh_1_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = np.cos(2.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))
        values[1] = np.sin(4.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))

    def value_shape(self):
        return (2,)
    
fsp.v_mesh_1.interpolate(v_mesh_1_Expression(element=fsp.V_mesh_1.ufl_element()))


msh.transfer_line_to_circle(fsp.v_mesh_1, fsp.v_sub_mesh_0_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])

io.full_print(fsp.v_sub_mesh_0_1, f'v_2d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

io.full_print(fsp.v_mesh_1, f'v_line', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
    
print(f'Comparing the two functions on polygon vertices: ')
error = 0

for i in range(rmsh.lmsh.mesh_parameters[0]['N']):
    print(f'v_line = {fsp.v_mesh_1(i*delta_l)}\t v_2d = {fsp.v_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))}')

    a = fsp.v_mesh_1(i*delta_l)
    b = fsp.v_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))

    for j in range(len(a)):
        if abs(a[j]-b[j]) > error:
            error = abs(a[j]-b[j])

print(f'error = {error}')



# 3 transfer tensor

# 3.1 transfer from 2d to line mesh

class t_sub_mesh_0_1_Expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
        # test case 1
        values[0] = np.cos(2* np.pi * (x[0]*x[1]))
        values[1] = np.sin(2* np.pi * x[0]) - np.sin(2* np.pi * x[1])
        values[2] = (np.sin(2* np.pi * x[0]) - np.sin(2* np.pi * x[1]))**2
        values[3] = np.cos(2* np.pi * x[0])**2 - np.sin(2* np.pi * x[1])
        values[4] = np.cos(2* np.pi * x[0])**3 - np.sin(2* np.pi * (x[0]+x[1]))
        values[5] = (np.cos(2* np.pi * x[0])**3 - np.sin(2* np.pi * (x[0]+x[1])))**2

    def value_shape(self):
        return (2, 3)
    

fsp.t_sub_mesh_0_1.interpolate(t_sub_mesh_0_1_Expression(element=fsp.T_sub_mesh_0_1.ufl_element()))


msh.transfer_circle_to_line(fsp.t_sub_mesh_0_1, fsp.t_mesh_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])

io.full_print(fsp.t_sub_mesh_0_1, f't_2d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

io.full_print(fsp.t_mesh_1, f't_line', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
    
print(f'Comparing the two functions on polygon vertices: ')
error = 0
for i in range(rmsh.lmsh.mesh_parameters[0]['N']):
    print(f't_line = {fsp.t_mesh_1(i*delta_l)}\t t_2d = {fsp.t_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))}')

    a = fsp.t_mesh_1(i*delta_l)
    b = fsp.t_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))

    for j in range(len(a)):
        if abs(a[j]-b[j]) > error:
            error = abs(a[j]-b[j])

print(f'error = {error}')


# 3.2 transfer from line mesh to 2d mesh 
class t_mesh_1_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = np.cos(2.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))
        values[1] = np.sin(4.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))
        values[2] = np.sin(2.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))**2
        values[3] = np.sin(4.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))**2
        values[4] = np.sin(6.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))
        values[5] = np.sin(6.0*np.pi*x[0]/(rmsh.lmsh.mesh_parameters[1]['x_r']-rmsh.lmsh.mesh_parameters[1]['x_l']))**2

    def value_shape(self):
        return (6,)
    
fsp.t_mesh_1.interpolate(t_mesh_1_Expression(element=fsp.T_mesh_1.ufl_element()))


msh.transfer_line_to_circle(fsp.t_mesh_1, fsp.t_sub_mesh_0_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])

io.full_print(fsp.t_sub_mesh_0_1, f't_2d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

io.full_print(fsp.t_mesh_1, f't_line', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
    
print(f'Comparing the two functions on polygon vertices: ')
error = 0
for i in range(rmsh.lmsh.mesh_parameters[0]['N']):
    print(f't_line = {fsp.t_mesh_1(i*delta_l)}\t t_2d = {fsp.t_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))}')

    a = fsp.t_mesh_1(i*delta_l)
    b = fsp.t_sub_mesh_0_1(np.add(rmsh.lmsh.parameters["c_r"], [rmsh.lmsh.parameters["r"] * np.cos(i * delta_theta), rmsh.lmsh.parameters["r"] * np.sin(i * delta_theta)]))

    for j in range(len(a)):
        if abs(a[j]-b[j]) > error:
            error = abs(a[j]-b[j])

print(f'error = {error}')
####################
'''



'''
here J[i][j] is the Jacobian of the functional for the j-th submesh of the i-th mesh, and similarly for problem, solver, ... 
'''

''''
The three variational problems (VPs) are solved as follows:
1)  Solve Poisson VP on sub_mesh[0][1] for u[0][1] ->
    Obtain 
        a) For test case 1: u[0][1] = (x[0] - cr[0]) + 2 * (x[1] - cr[1])
        b) For test case 2: u[0][1] = 2 * (x[0] - cr[0])**3 + (x[1] - cr[1])**3

2)  Transfer u[0][1] on mesh[1] -> u_0_1_on_1
    r theta = s
    L1 = 2 pi r 
    Given that 
        x[0] = cr[0] + r * cos(s/r), 
        x[1] = cr[1] + r * sin(s/r), 
        where s is the coordinate along mesh[1] and along the circle, we have 

        a) For test case 1: u_0_1_on_1(s)  = (r * cos(s/r)) + 2 * (r * sin(s/r))
        b) For test case 2: u_0_1_on_1(s)  = 2 * (r * cos(s/r))**3 + (r * sin(s/r))**3

3)  Solve on mesh[1] the VP

        u[1]'(s) = u_0_1_on_1(s)

    The solution is 

      a) For test case 1: u[1](s) = C[1] + r^2 (-2 Cos[s/r] + Sin[s/r])
      b) For test case 2: u[1](s) = C[1] + 1/12 r^4 (-9 Cos[s/r] + Cos[(3 s)/r] + 4 (5 + Cos[(2 s)/r]) Sin[s/r])
    
    where I set C[1] -> 0 by adding a Dirichlet BC on the VP on mesh[1]


4)  Transfer u[1](s) to sub_mesh[0][0] and write it in u_1_on_0_0. 
    On the circle 

        a) Test case 1: u_1_on_0_0 = - 2 r * (x[0] - cr[0]) + r * (x[1] - cr[1])
        b) Test case 2: u_1_on_0_0 =  1/12 r (9 r^2 (crx - x) + (-crx + x)^3 + 3 (crx - x) (cry - y)^2 + 
   2 (cry - y)^3 + 18 r^2 (-cry + y) + 6 (crx - x)^2 (-cry + y))

5)  Solve a Poisson problem on sub_mesh[0][0]  with BC on the circle u[0][0] = u_1_on_0_0

    The problem has exact solution 
        a) Test case 1:  u[0][0] = - 2 r * (x[0] - cr[0]) + r * (x[1] - cr[1]),  
        b) Test case 2:  u[0][0] = 1/12 r (9 r^2 (crx - x) + (-crx + x)^3 + 3 (crx - x) (cry - y)^2 + 
   2 (cry - y)^3 + 18 r^2 (-cry + y) + 6 (crx - x)^2 (-cry + y))

    Obtain 
        a) Test case 1): u[0][0] = - 2 r * (x[0] - cr[0]) + r * (x[1] - cr[1]) in sub_mesh[0][0]
        b) Test case 2):  u[0][0] = 1/12 r (9 r^2 (crx - x) + (-crx + x)^3 + 3 (crx - x) (cry - y)^2 + 
   2 (cry - y)^3 + 18 r^2 (-cry + y) + 6 (crx - x)^2 (-cry + y))
'''


J, problem, solver, vp = [[None]*2, None], [[None]*2, None], [[None]*2, None], [[None]*2, None]

# solve the variational problem in sub_mesh[0][1], and obtain the solution 
print('Solving the problem in sub_mesh[0][1]...')
vp[0][1] = importlib.import_module(swi.vp_sub_mesh_0_1)
J[0][1] = derivative(vp[0][1].F, fsp.u[0][1], fsp.J_u[0][1])
problem[0][1] = NonlinearVariationalProblem(vp[0][1].F, fsp.u[0][1], vp[0][1].bcs, J[0][1])
solver[0][1] = NonlinearVariationalSolver(problem[0][1])

solver[0][1].solve()
print('...done.')

'''
print(f'Transferring solution on sub_mesh[0][1] to mesh[1] ...')
msh.transfer_circle_to_line(fsp.u[0][1], fsp.u_0_1_on_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])
print(f'... done.')


io.full_print(fsp.u_0_1_on_1, f'u_0_1_on_1', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)



# solve the variational problem on mesh[1]
print('Solving the problem in mesh[1]...')
# use the solution obtained for sub_mesh[0][1] in the variational problem on mesh[1]
vp[1] = importlib.import_module(swi.vp_mesh_1)
J[1] = derivative(vp[1].F, fsp.u[1], fsp.J_u[1])
problem[1] = NonlinearVariationalProblem(vp[1].F, fsp.u[1], vp[1].bcs, J[1])
solver[1] = NonlinearVariationalSolver(problem[1])

solver[1].solve()
print('...done.')

print(f'Transferring solution on mesh[1] to sub_mesh[0][0] ...')
msh.transfer_line_to_circle(fsp.u[1], fsp.u_1_on_0_0, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])
print(f'... done.')


vp[0][0] = importlib.import_module(swi.vp_sub_mesh_0_0)
J[0][0] = derivative(vp[0][0].F, fsp.u[0][0], fsp.J_u[0][0])
problem[0][0] = NonlinearVariationalProblem(vp[0][0].F, fsp.u[0][0], vp[0][0].bcs, J[0][0])
solver[0][0] = NonlinearVariationalSolver(problem[0][0])

print('Solving the problem in sub_mesh[0][0]...')
solver[0][0].solve()
print('...done.')


prout_bc = importlib.import_module(swi.prout_bc)
'''