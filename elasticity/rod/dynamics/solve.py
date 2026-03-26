'''
This code solves for the dynamics of the deformation field of an elastic rod subjected to gravity

Run with
    python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/elasticity/rod/dynamics/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_a $MESH_PATH $SOLUTION_PATH
'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr

prout = importlib.import_module(swi.prout)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

# dolfin.parameters["form_compiler"]["quadrature_degree"] = 10


print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + "metadata.csv", \
                                io.merge_dictionaries(rmsh.parameters, rpam.parameters))


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

# set the initial profiles
class u_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = - x[0]/rmsh.parameters['L'] * rmsh.parameters['h']

    def value_shape(self):
        return (2,)

class v_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)

fsp.u_n_1.interpolate(u_0_expression(element=fsp.U_u_n.ufl_element()))
fsp.v_n_1.interpolate(v_0_expression(element=fsp.U_v_n.ufl_element()))

print("Starting time iteration ...", flush=True)

# Time-stepping
t = 0
step = 0

for n in range(rpam.parameters['num_steps']):
    # Update current time
    t += vp.dt
    step += 1

    vp = importlib.import_module(swi.vp)

    var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi, parameters=params)

    u_n_output, v_n_output = fsp.psi.split( deepcopy=True)
    fsp.u_n_1.assign(u_n_output)
    fsp.v_n_1.assign(v_n_output)

    prout.print_bcs(fsp.psi)
    
    if step % rpam.parameters['print_out_stride'] == 0:
        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        prout.print_solution(fsp.psi, step, t)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)
