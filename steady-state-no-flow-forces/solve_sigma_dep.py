'''
This file solves for the steady state of a two-dimensional fluid with no flows

Run with
clear; python3 solve.py [path where to read the mesh] [path where to store the solution]

Note that all sections of the code which need to be changed when an external parameter (e.g. the length of the Rectangle, etc...) is changed are bracketed by
#CHANGE PARAMETERS HERE
'''

import colorama as col
from fenics import *
from mshr import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv

#add the path where to find the shared modules
module_path = '/home/tanos/Thesis/Fenics-files-for-thesis/modules'
sys.path.append(module_path)

from mesh import mesh as mesh_module
from function_spaces import fsp
from variational_problem_bc_ring import vp
import input_output as io
import physics as phys
from runtime_arguments import rarg
from geometry import geo
from boundary_geometry import bgeo

import ufl as ufl

from read_mesh_ring import rmsh
mesh_module = mesh_module()

j = ufl.indices( 1 )

set_log_level( 20 )
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

rarg = rarg()

geo = geo()
bgeo = bgeo(mesh_module, geo, rarg.args.input_directory)
rmsh = rmsh(mesh_module, rarg.args, bgeo, geo)
#rmsh.test_mesh()
fsp = fsp(bgeo)
vp = vp(geo, fsp, bgeo, rmsh)
print("Input directory = ", rarg.args.input_directory )
print("Output directory = ", rarg.args.output_directory )
print(f"Radius of mesh cell = {col.Fore.CYAN}{rmsh.r_mesh}{col.Style.RESET_ALL}")

# CHANGE PARAMETERS HERE

# solve the variational problem




z_R_const = 0.0
z_r_const = 0.0
omega_r = -0.1
omega_R = 0.0


#set the solver parameters here
params = {'nonlinear_solver': 'newton',
           'newton_solver':
            {
                'linear_solver'           : 'superlu',
                # 'linear_solver'           : 'mumps',
                # 'linear_solver':   'lu',
                'absolute_tolerance'      : 1e-6,
                'relative_tolerance'      : 1e-6,
                'maximum_iterations'      : 1000000,
                'relaxation_parameter'    : 0.95,
             }
}

omega_rs = [-0.1, 0.0, 0.1]
ks = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0]
z_rs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


variable = io.variables(["k", "z_r", "h", "H", "w", "Fz"], rarg.args.output_directory)

for k in ks:
    vp.define_variational_problem(k, z_R_const, z_r_const, omega_r, omega_R)
    for omega_r in omega_rs:
        vp.change_nitsche_bc(omega_r, omega_R)
        for z_r in z_rs:
            z_r_const = z_r
            vp.change_dirichlet_bc(z_r_const, z_R_const)
            J = derivative( vp.F, fsp.psi, fsp.J_psi )
            problem = NonlinearVariationalProblem( vp.F, fsp.psi, vp.bcs, J )
            solver = NonlinearVariationalSolver( problem )
            solver.parameters.update(params)
            solver.solve()

            z_output, omega_output, mu_output = fsp.psi.split( deepcopy=True )
            h_ = assemble(z_output * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)/assemble(Constant(1) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)
            H_ = assemble(mu_output * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)/assemble(Constant(1) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)
            w_ = assemble(((bgeo.n_circle( omega_output ))[j] * omega_output[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)/assemble(Constant(1) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r) 
            Fz_t = assemble(geo.from_t_to_3D(omega_output, phys.force_el_sigma_t(mu_output, k, fsp.sigma, bgeo.n_circle(omega_output)))[2]* bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)
            variable.add([k, z_r, h_, H_, w_, Fz_t])

variable.print_to_csvfile(rarg.args.output_directory + "/variables.csv")

'''
#nh is the vector field of the normal to the boundary extended to the bulk
nh = bgeo.calc_normal_cg2(omega_output, bgeo.mesh) #PROJECTION OF NH CREATES PROBLEMS SO SHOULD BE AVOIDED
e = project(geo.e(omega_output), fsp.Q_e)
print("e = ", e(0,1), e(1,0))
print("nh = ", nh(0,1), nh(1,0))
F_m = project((-1)*phys.force_el_modulus(mu_output, kappa, fsp.sigma), fsp.Q_F)
F_el_sigma_3D = lambda x,y : [F_m(x,y)*(e(x,y)[i]*nh(x,y)[0]+e(x,y)[3+i]*nh(x,y)[1]) for i in range(3)]
#Fz_ = assemble(F_el_sigma_3D[2]* bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)
'''
'''
h = np.array(h).reshape(-1, 1)
H = np.array(H)
w = np.array(w)
Fz = np.array(Fz)

model1 = LinearRegression()
model1.fit(h, w)
slope1 = model1.coef_[0]  # Slope
intercept1 = model1.intercept_  # Intercept

model2 = LinearRegression()
model2.fit(h, Fz)
slope2 = model2.coef_[0]  # Slope
intercept2 = model2.intercept_  # Intercept

model3 = LinearRegression()
model3.fit(h, H)
slope3 = model3.coef_[0]  # Slope
intercept3 = model3.intercept_  # Intercept



print("w = ", slope1, "h + ", intercept1)
print("Fz = ", slope2, "h + ", intercept2)
print("H = ", slope3, "h + ", intercept3)

io.print_vector_boundary_to_csvfile(F_el_sigma_3D, bgeo.mesh, "output/force_vector.csv")
# Predictions
H_pred = model3.predict(h)
Fz_pred = model2.predict(h)
w_pred = model1.predict(h)
fig, ax = plt.subplots(3, 1)
'''
'''p1 = ax[0].scatter(h, w, color='blue')
p2 = ax[0].plot(h, w_pred, color='red')
ax[0].set_xlabel('h')
ax[0].set_ylabel('w')
ax[0].legend(['w', 'w_pred'])'''

'''
p3 = ax[1].scatter(h, Fz, color='blue')
p4 = ax[1].plot(h, Fz_pred, color='red')
ax[1].set_xlabel('h')
ax[1].set_ylabel('Fz')
ax[1].legend(['Fz', 'Fz_pred'])


p3 = ax[2].scatter(h, H, color='green')
p4 = ax[2].plot(h, H_pred, color='red')
ax[2].set_xlabel('h')
ax[2].set_ylabel('H')
ax[2].legend(['H'])

print(h, H, w, Fz)
#plt.show()'''