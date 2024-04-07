import numpy as np
from fenics import *
from mshr import *
import numpy as np
import meshio
import ufl as ufl

#r, R must be the same as in generate_mesh.py
R = 1.0
r = 0.25
#these must be the same c_R, c_r as in generate_mesh.py, with the third component dropped
c_R = [0.0, 0.0]
c_r = [0.0, -0.1]


# Define norm of x
def norm(x):
    return (np.sqrt(x[0]**2 + x[1]**2))



#analytical expression for a vector
class MyVectorFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]
        values[1] = -x[1]
    def value_shape(self):
        return (2,)
#analytical expression for a function
class MyScalarFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
    def value_shape(self):
        return (1,)
t=0