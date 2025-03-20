from fenics import *
from mshr import *
import ufl as ufl
import numpy as np

import function_spaces as fsp
import geometry as geo
import read_mesh_bc_obstacle as rmsh
import runtime_arguments as rarg

i, j, k, l = ufl.indices( 4 )

# CHANGE PARAMETERS HERE
T = (float)( rarg.args.T )
num_steps = (int)( rarg.args.N )

dt = T / num_steps  # time step size
rho = 1.0
mu = 0.001


# CHANGE PARAMETERS HERE


# trial analytical expression for a vector
class TangentVelocityExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)


class OmegaExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = np.cos( 2.0 * np.pi * x[0] )
        values[1] = x[1]

    def value_shape(self):
        return (2,)


# trial analytical expression for the  surface tension sigma(x,y)
class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        # values[0] = 4*x[0]*x[1]*sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
        # values[0] = cos(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
        values[0] = 0.0

    def value_shape(self):
        return (1,)


# trial analytical expression for w
class NormalVelocityExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0

    def value_shape(self):
        return (1,)


v__profile_l = Expression( ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(h, 2)', '0'), degree=2, h=rmsh.h )

bc_v__inflow = DirichletBC( fsp.Q_v, v__profile_l, rmsh.inflow )
bc_v__walls = DirichletBC( fsp.Q_v, Constant( (0, 0) ), rmsh.walls )
bc_v__cylinder = DirichletBC( fsp.Q_v, Constant( (0, 0) ), rmsh.cylinder )

bc_phi_outflow = DirichletBC( fsp.Q, Constant( 0 ), rmsh.outflow )

# boundary conditions for the surface_tension p
bc_v_ = [bc_v__walls, bc_v__inflow, bc_v__cylinder]
bc_phi = [bc_phi_outflow]

# Define variational problem for step 1
# step 1 for v
F1 = ( \
                 rho * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                        + (3.0 / 2.0 * fsp.v_n_1[j] - 1.0 / 2.0 * fsp.v_n_2[j]) * geo.Nabla_v( fsp.V, fsp.omega )[i, j]) * fsp.nu[i] \
                 + fsp.sigma_n_32 * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu, fsp.omega )[i, j] + 2.0 * mu * geo.d_c( fsp.V, fsp.w, fsp.omega )[j, i] * geo.Nabla_f( fsp.nu, fsp.omega )[j, i] \
         ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

# step 2
F2 = (geo.g_c( fsp.omega )[i, j] * (fsp.phi.dx( i )) * (fsp.q.dx( j )) + (rho / dt) * (geo.Nabla_v( fsp.v_, fsp.omega )[i, i]) * fsp.q) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

# Define variational problem for step 3
F3 = (((fsp.v_n[i] - fsp.v_[i]) + (dt / rho) * geo.g_c( fsp.omega )[i, j] * (fsp.phi.dx( j ))) * fsp.nu[i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx
