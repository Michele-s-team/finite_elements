from fenics import *
import dolfin
import ufl as ufl

import physics.elasticity as ela


alpha, beta, gamma, delta = ufl.indices(4)

'''
diffusive current in the ALE formulation
Input values:
    - `u`:  displacement field
    - `D`: \cal{D} in  fluid_structure_interation/fluid_obstacle/notes, diffusion coefficient in reference coordinates
    - `G`: inverse of deformation-gradient tensor
    - `v`: \rm{v} in fluid_structure_interation/fluid_obstacle/notes, advecting velocity in reference coordinates

Return values; 
    - \cal{J}^M_alpha in fluid_structure_interation/fluid_obstacle/notes
'''
def J_ale(u, c, v, D):

    return as_tensor(-D*ela.G(u)[beta, alpha]*(c.dx(beta)) + v[alpha]*c, (alpha))
