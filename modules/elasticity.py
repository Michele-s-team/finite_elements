from fenics import *
import dolfin
import ufl as ufl

import geometry as geo

i, j, k, l = ufl.indices(4)

def test(u):
    num_components = u.function_space().num_sub_spaces()
    print(f"Number of components: {num_components}")