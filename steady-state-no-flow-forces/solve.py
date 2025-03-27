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

#add the path where to find the shared modules
module_path = '/home/tanos/Thesis/Fenics-files-for-thesis/modules'
sys.path.append(module_path)

import function_spaces as fsp
import variational_problem_bc_ring as vp
import input_output as io
import physics as phys
import runtime_arguments as rarg
import geometry as geo
import boundary_geometry as bgeo
import ufl as ufl

import read_mesh_ring as rmsh

j = ufl.indices( 1 )

set_log_level( 20 )
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

print("Input diredtory = ", rarg.args.input_directory )
print("Output diredtory = ", rarg.args.output_directory )
print(f"Radius of mesh cell = {col.Fore.CYAN}{rmsh.r_mesh}{col.Style.RESET_ALL}")

# CHANGE PARAMETERS HERE

# solve the variational problem

h = []
Fz = []
H = []
w = [ ]
z_R_const = 0.0
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

for i in range(1,2):
    # Reset psi before solving
    #fsp.psi.assign(Function(fsp.Q))  # Reset psi
    
    z_r_const = 0.05*i
    bc_z_r = DirichletBC( fsp.Q.sub( 0 ), z_r_const, rmsh.boundary_r )
    bc_z_R = DirichletBC( fsp.Q.sub( 0 ), z_R_const, rmsh.boundary_R )
    # CHANGE PARAMETERS HERE

    # all BCs
    bcs = [bc_z_r, bc_z_R]
    J = derivative( vp.F, fsp.psi, fsp.J_psi )
    problem = NonlinearVariationalProblem( vp.F, fsp.psi, bcs, J )
    solver = NonlinearVariationalSolver( problem )
    solver.parameters.update(params)
    solver.solve()

    z_output, omega_output, mu_output = fsp.psi.split( deepcopy=True )
    h_ = assemble(z_output * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)/assemble(Constant(1) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)
    H_ = assemble(mu_output * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)/assemble(Constant(1) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)
    w_ = assemble(((bgeo.n_circle( omega_output ))[j] * omega_output[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)/assemble(Constant(1) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r) 
    Fz_ = assemble(geo.from_t_to_3D(omega_output, phys.force_el_sigma_t(mu_output, vp.kappa, fsp.sigma, bgeo.n_circle(omega_output)))[2]* bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r)* rmsh.ds_r)
    h.append(h_)
    H.append(H_)
    w.append(w_)
    Fz.append(Fz_)
    print("h = ", h_, " instead of ", z_r_const)
    print("H = ", H_)
    print("w = ", w_)
    print("Fz = ", Fz_)


#nh is the vector field of the normal to the boundary extended to the bulk
nh = bgeo.calc_normal_cg2(bgeo.mesh)
print("nh = ", geo.from_t_to_3D(omega_output, phys.force_el_sigma_t(mu_output, vp.kappa, fsp.sigma, geo.norm_t(omega_output, geo.from_3D_to_t(omega_output, nh)))))
nh = as_tensor([nh[0], nh[1], 0.0])
F_el_sigma_3D = [project(geo.from_t_to_3D(omega_output, phys.force_el_sigma_t(mu_output, vp.kappa, fsp.sigma, geo.norm_t(omega_output, geo.from_3D_to_t(omega_output, nh))))[i], fsp.Q_F) for i in range(3)]


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

'''p1 = ax[0].scatter(h, w, color='blue')
p2 = ax[0].plot(h, w_pred, color='red')
ax[0].set_xlabel('h')
ax[0].set_ylabel('w')
ax[0].legend(['w', 'w_pred'])'''

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
plt.show()

'''
# Create XDMF files for visualization output
xdmffile_z = XDMFFile( (rarg.args.output_directory) + '/z.xdmf' )
xdmffile_omega = XDMFFile( (rarg.args.output_directory) + '/omega.xdmf' )
xdmffile_mu = XDMFFile( (rarg.args.output_directory) + '/mu.xdmf' )

xdmffile_nu = XDMFFile( (rarg.args.output_directory) + '/nu.xdmf' )
xdmffile_tau = XDMFFile( (rarg.args.output_directory) + '/tau.xdmf' )

xdmffile_sigma = XDMFFile( (rarg.args.output_directory) + '/sigma.xdmf' )

xdmffile_f = XDMFFile( (rarg.args.output_directory) + '/f.xdmf' )
xdmffile_f.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output = fsp.psi.split( deepcopy=True )



# print solution to file
xdmffile_z.write( z_output, 0 )
xdmffile_omega.write( omega_output, 0 )
xdmffile_mu.write( mu_output, 0 )

xdmffile_nu.write( fsp.nu, 0 )
xdmffile_tau.write( fsp.tau, 0 )

xdmffile_sigma.write( fsp.sigma, 0 )

io.print_scalar_to_csvfile(z_output, (rarg.args.output_directory) + '/z.csv')
io.print_vector_to_csvfile(omega_output, (rarg.args.output_directory) + '/omega.csv')
io.print_scalar_to_csvfile(mu_output, (rarg.args.output_directory) + '/mu.csv')

io.print_vector_to_csvfile(fsp.nu, (rarg.args.output_directory) + '/nu.csv')
io.print_scalar_to_csvfile(fsp.tau, (rarg.args.output_directory) + '/tau.csv')

io.print_scalar_to_csvfile(fsp.sigma, (rarg.args.output_directory) + '/sigma.csv')


# write the solutions in .h5 format so it can be read from other codes
HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/h5/z.h5", "w" ).write( z_output, "/f" )
HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/h5/omega.h5", "w" ).write( omega_output, "/f" )
HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/h5/mu.h5", "w" ).write( mu_output, "/f" )

HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/h5/nu.h5", "w" ).write( fsp.nu, "/f" )
HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/h5/tau.h5", "w" ).write( fsp.tau, "/f" )

HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/h5/sigma.h5", "w" ).write( fsp.sigma, "/f" )

xdmffile_f.write( project(phys.fel_n( omega_output, mu_output, fsp.tau, vp.kappa ), fsp.Q_sigma), 0 )
xdmffile_f.write( project(-phys.flaplace( fsp.sigma, omega_output), fsp.Q_sigma), 0 )

# import print_out_bc_square_a
#import print_out_bc_square_b
import print_out_bc_ring
# import print_out_bc_square_no_circle_a'
'''