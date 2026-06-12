from fenics import *
import importlib
import ufl as ufl


import command as cmd
import differential_geometry.manifold.geometry as geo
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('monge')

i, j, k, l = ufl.indices( 4 )


dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size


#trial analytical expression for a vector
class TangentVelocityExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

class ManifoldExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 2 * x[1] * (rmsh.parameters["h"] - x[1]) / rmsh.parameters["h"]**2 * (x[1] - rmsh.parameters["h"] / 24) / rmsh.parameters["h"]
    def value_shape(self):
        return (1,)

class OmegaExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = -((rmsh.parameters["h"]**2) - 50.0*rmsh.parameters["h"]*x[1] + 72.0*((x[1])**2))/(12.0*rmsh.parameters["h"]**3)
    def value_shape(self):
        return (2,)

# trial analytical expression for the  surface tension sigma(x,y)
class SurfaceTensionExpression(UserExpression):
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



v__profile_l = Expression( ('4.0*1.5*x[1]*(h - x[1])', '0'), degree=2, h=rmsh.parameters["h"] )

bc_v__inflow = DirichletBC( fsp.Q_v, v__profile_l, rmsh.boundary_l )
bc_v__walls = DirichletBC( fsp.Q_v, Constant( (0, 0) ), rmsh.boundary_tb )

bc_phi_outflow = DirichletBC( fsp.Q, Constant( 0 ), rmsh.boundary_r )

# boundary conditions for the surface_tension p
bc_v_ = [bc_v__walls, bc_v__inflow]
bc_phi = [bc_phi_outflow]


# Define variational problem for step 1
# step 1 for v
F1 = ( \
                 rpam.parameters['rho'] * ((fsp.v_[i]- fsp.v_n_1[i]) / dt \
                        + (3.0/2.0 * fsp.v_n_1[j] - 1.0/2.0 * fsp.v_n_2[j]) * geo.Nabla_v( fsp.V, fsp.omega )[i, j]) * fsp.nu[i] \
                 + fsp.sigma_n_32 * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu, fsp.omega )[i, j] + 2.0 * rpam.parameters['mu'] * geo.d_c( fsp.V, fsp.w, fsp.omega )[j, i] * geo.Nabla_f( fsp.nu, fsp.omega )[j, i]  \
    ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

# step 2
F2 = ( geo.g_c( fsp.omega )[i, j] * (fsp.phi.dx( i )) * (fsp.q.dx( j )) + (rpam.parameters['rho'] / dt) * (geo.Nabla_v( fsp.v_, fsp.omega )[i, i]) * fsp.q ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

# Define variational problem for step 3
F3 = ( ((fsp.v_n[i] - fsp.v_[i]) + (dt/rpam.parameters['rho']) * geo.g_c( fsp.omega )[i, j] * (fsp.phi.dx( j ))) * fsp.nu[i] ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx