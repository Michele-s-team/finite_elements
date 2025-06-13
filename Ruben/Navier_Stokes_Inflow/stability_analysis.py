#!/usr/bin/env python3
# solve.py

import sys
if len(sys.argv) != 4:
    print("Usage: solve.py <problem_name> <mesh_input_dir> <solution_output_dir>")
    sys.exit(1)

problem_name, mesh_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

# 1) Tell runtime_arguments (used by the variational module) where to find mesh & outputs
import runtime_arguments as rarg
rarg.args.input_directory   = mesh_dir
rarg.args.output_directory  = out_dir

# 2) Import your problem module
import importlib
modname = f"variational_problem_bc_{problem_name}"
vp      = importlib.import_module(modname)

# 3) Build & solve the nonlinear problem
from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
problem = NonlinearVariationalProblem(vp.F, vp.psi, vp.bcs, vp.J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["linear_solver"]      = "mumps"
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-6
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6
solver.parameters["newton_solver"]["maximum_iterations"] = 50
solver.solve()

# 4) Split out the components
u_sub, p_sub = vp.psi.split()

# 5) Project onto standalone spaces so CSV export works
from dolfin import VectorFunctionSpace, FunctionSpace, project
V = VectorFunctionSpace(vp.mesh, "Lagrange", 2)
u = project(u_sub, V)
Q = FunctionSpace(vp.mesh, "Lagrange", 1)
p = project(p_sub, Q)

# 6) Write all outputs (XDMF, H5, CSV, nodal CSV)
import input_output as io
io.full_print(u, "velocity",
              out_dir, out_dir, out_dir, out_dir + "/nodal_values", vp.mesh, "vector")
io.full_print(p, "pressure",
              out_dir, out_dir, out_dir, out_dir + "/nodal_values", vp.mesh, "scalar")


import numpy as np, matplotlib.tri as tri
import matplotlib.pyplot as plt

# extract vertices & velocity components
coords = vp.mesh.coordinates()
cells  = vp.mesh.cells()
uv     = u.compute_vertex_values(vp.mesh)
ux, uy  = uv[0::2], uv[1::2]

triang = tri.Triangulation(coords[:,0], coords[:,1], cells)
plt.figure(figsize=(6,6))
plt.tricontourf(triang, p.vector().get_local(), levels=30)  # background pressure
plt.colorbar(label="p")
plt.quiver(coords[:,0], coords[:,1], ux, uy, scale=5, color="k")
plt.title("Pressure & Velocity")
plt.axis("equal")
plt.show()
