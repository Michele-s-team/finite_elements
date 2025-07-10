#!/usr/bin/env python3
import importlib
from runtime_arguments import args
import matplotlib.pyplot as plt
import numpy as np
import dolfin
from fenics import *
from dolfin import Function, assemble, dot, sqrt, Function
import function_spaces as fsp
import print_out_solution as pr_sol
import runtime_arguments as rarg
import switch_problem as swi
def run_one(Re_val):
    rarg.args.Re = Re_val

    # 2) reload any modules that read args.Re at import-time
    import variational_problem_bc_square as vp_mod
    import solve_steady as ss_mod
    importlib.reload(vp_mod)
    importlib.reload(ss_mod)
    vp_mod  = importlib.reload(importlib.import_module(swi.vp))
    prbc_mod = importlib.reload(importlib.import_module(swi.prout_bc))
    print(f"\n=== Running transient at Re = {Re_val:.3g} ===")
    print("Re = ", vp_mod.Re)
    # 3) solve the steady state
    rmsh, _, u_star, p_star = ss_mod.solve_steady()
    steady_u = Function(fsp.Q_v)
    steady_u.interpolate(u_star)
    steady_p = Function(fsp.Q)
    steady_p.interpolate(p_star)

    # 5) Seed pressure fields
    fsp.sigma_n_32.assign(steady_p)   # used as p^n
    fsp.phi.assign(steady_p)          # used as p^{n+1}
    # 6) Add a small perturbation to velocity
    eps = 1e-3
    perturb = Expression(
        ("eps*sin(2*pi*x[0])", "eps*sin(2*pi*x[1])"),
        degree=2, eps=eps
    )
    u_pert = project(perturb, fsp.Q_v)
    steady_u.vector().axpy(1.0, u_pert.vector())  # steady_u += u_pert
    u0_diff = steady_u - u_star
    err0 = sqrt(assemble(dot(u0_diff, u0_diff)*rmsh.dx))
    # 7) Initialize transient velocities
    fsp.v_n_1.assign(steady_u)
    fsp.v_n_2.assign(steady_u)

    # 8) Time‐stepping loop
    t = 0.0
    step = 0
    for n in range(vp_mod.num_steps):
        t += vp_mod.dt
        step += 1

        # reload vp so it sees the new Re if needed
        vp_mod = importlib.reload(importlib.import_module(swi.vp))

        # step 1: momentum
        J1 = derivative(vp_mod.F1, fsp.v_,   fsp.J_v_)
        prob1 = NonlinearVariationalProblem(vp_mod.F1, fsp.v_,   vp_mod.bc_v_, J1)
        sol1  = NonlinearVariationalSolver(prob1); sol1.solve()

        # step 2: pressure Poisson
        J2 = derivative(vp_mod.F2, fsp.phi, fsp.J_phi)
        prob2 = NonlinearVariationalProblem(vp_mod.F2, fsp.phi, vp_mod.bc_phi, J2)
        sol2  = NonlinearVariationalSolver(prob2); sol2.solve()

        # step 3: velocity correction
        J3 = derivative(vp_mod.F3, fsp.v_n, fsp.J_v_n)
        prob3 = NonlinearVariationalProblem(vp_mod.F3, fsp.v_n, [], J3)
        sol3  = NonlinearVariationalSolver(prob3); sol3.solve()

        # print boundary‐condition info
        prbc_mod.print_bcs()

        # update σ and φ (pressure history)
        fsp.sigma_n_12.assign(fsp.sigma_n_32 - fsp.phi)
        fsp.sigma_n_32.assign(fsp.sigma_n_12)

        # update velocity history
        fsp.v_n_2.assign(fsp.v_n_1)
        fsp.v_n_1.assign(fsp.v_n)

        # output solution at this step
        pr_sol.print_solution(t, step, vp_mod.dt)
        print(f"\t{100.0*(t/vp_mod.T):.2f} %", flush=True)

    # 6) at the end: compute L2‐errors
    u_diff = fsp.v_n - steady_u
    err_u  = float(sqrt(assemble(dot(u_diff, u_diff)*rmsh.dx)))
    print("\n--- Deviation from steady state ---")
    print(f"||u(T)-u_*||_L2 = {err_u:.6e}")
    print("-----------------------------------")   
    return err_u/err0
if __name__ == "__main__":
    # list of Re to sweep
    Re_values = np.logspace(-2, 2, 20)   

    errs_u = []
    for Re_val in Re_values:
        print(f"→ Running at Re = {Re_val:.3g}")
        eu = run_one(Re_val)
        print(f"   ‖u−u_*‖ = {eu:.2e}")
        errs_u.append(eu)

    # plot at the end
    plt.loglog(Re_values, errs_u, "o-", label="velocity error")
    plt.xlabel("Reynolds number Re")
    plt.ylabel("L² deviation from steady")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("velocity_error_vs_Re_1.png")
    plt.show()
