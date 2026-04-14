'''
this example shows the correct and wrong approach to define a step function with a discontinuity, which is supposed to be used in a variational forms 
'''

from fenics import *

mesh = UnitSquareMesh(4, 4)
Q = FunctionSpace(mesh, "DG", 1)

# ── UserExpression approach (wrong at interface DOFs) ────────────────────────
class f_wrong(UserExpression):
    def eval(self, values, x):
        if x[0] < 0.5:
            values[0] = 1.0
        else:
            values[0] = 0.0
    def value_shape(self):
        return (1,)

f1 = Function(Q)
f1.interpolate(f_wrong(element=Q.ufl_element()))

# ── UFL conditional approach (correct everywhere) ────────────────────────────
x_  = SpatialCoordinate(mesh)
f2  = conditional(le(x_[0], 0.5), 1.0, 0.0)

# ── Check: integrate f over left half — should give 0.5 * 1.0 = 0.5 ─────────
dx = Measure("dx", domain=mesh)

print(f"UserExpression integral = {assemble(f1 * dx):.6f}")   # may be wrong near x=0.5
print(f"conditional integral    = {assemble(f2 * dx):.6f}")   # always correct