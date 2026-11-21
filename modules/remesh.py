"""
remesh.py — place in /home/fenics/shared/modules/

Full remesh procedure for the free-boundary fluid problem.
Called from solve.py after each N_remesh timesteps.
"""

import os
import glob
import numpy as np
import gmsh
from fenics import *
from dolfin import LagrangeInterpolator
import importlib

import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


# -----------------------------------------------------------------------
def _extract_bottom_coords(mesh, mf, bottom_id):
    """Extract and x-sort the coordinates of bottom boundary nodes."""
    V = FunctionSpace(mesh, "CG", 1)
    v2d = vertex_to_dof_map(V)
    dofs = []
    for facet in facets(mesh):
        if mf[facet.index()] == bottom_id:
            for vertex in facet.entities(0):
                dofs.append(v2d[vertex])
    unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
    coords = V.tabulate_dof_coordinates()[unique_dofs]
    coords = coords[np.argsort(coords[:, 0])]
    return coords


# -----------------------------------------------------------------------
def build_new_mesh(bottom_coords, step, path_to_meshfiles,
                   L, h, gridsize, bottom_tag, top_tag, sides_tag):
    """
    Build a new gmsh mesh from the displaced bottom boundary coords.
    Returns xml_base (path WITHOUT .xml extension).
    """
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("domain_%i" % step)

    # bottom boundary: displaced spline
    bottom_pts = []
    for (x, y) in bottom_coords:
        p = gmsh.model.geo.addPoint(x, y, 0, gridsize)
        bottom_pts.append(p)

    # top boundary: flat at y=h
    p_tl = gmsh.model.geo.addPoint(0.0, h, 0, gridsize)
    p_tr = gmsh.model.geo.addPoint(L,   h, 0, gridsize)

    line_bottom = gmsh.model.geo.addSpline(bottom_pts)
    line_right  = gmsh.model.geo.addLine(bottom_pts[-1], p_tr)
    line_top    = gmsh.model.geo.addLine(p_tr, p_tl)
    line_left   = gmsh.model.geo.addLine(p_tl, bottom_pts[0])

    cl   = gmsh.model.geo.addCurveLoop([line_bottom, line_right, line_top, line_left])
    surf = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.addPhysicalGroup(2, [surf],                  tag=1)
    gmsh.model.geo.addPhysicalGroup(1, [line_bottom],           tag=bottom_tag)
    gmsh.model.geo.addPhysicalGroup(1, [line_top],              tag=top_tag)
    gmsh.model.geo.addPhysicalGroup(1, [line_left, line_right], tag=sides_tag)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [line_bottom])
    gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", 1000)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField",  1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin",  gridsize / 5)
    gmsh.model.mesh.field.setNumber(2, "SizeMax",  gridsize)
    gmsh.model.mesh.field.setNumber(2, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax",  h / 2)
    gmsh.model.mesh.field.add("Min", 7)
    gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(7)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)

    gmsh.model.mesh.generate(dim=2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    msh_file = path_to_meshfiles + "mesh_%i.msh" % step
    gmsh.write(msh_file)
    gmsh.finalize()

    xml_base = path_to_meshfiles + "domain"
    os.system("dolfin-convert %s %s.xml" % (msh_file, xml_base))

    # return base path WITHOUT .xml so caller can find both xml files
    return xml_base


# -----------------------------------------------------------------------
def _interp(old_func, new_space):
    """Interpolate old_func onto new_space using LagrangeInterpolator."""
    new_func = Function(new_space)
    LagrangeInterpolator.interpolate(new_func, old_func)
    return new_func


# -----------------------------------------------------------------------
def _load_facet_mf(new_mesh, xml_base):
    """
    Load the boundary MeshFunction produced by dolfin-convert.
    Tries both _facet_region.xml and _physical_region.xml naming conventions.
    """
    candidates = (
        glob.glob(xml_base + "_facet_region.xml") +
        glob.glob(xml_base + "_physical_region.xml") +
        glob.glob(os.path.dirname(xml_base) + "/*facet*.xml") +
        glob.glob(os.path.dirname(xml_base) + "/*physical*.xml")
    )
    if not candidates:
        raise RuntimeError(
            "No facet region XML found after dolfin-convert in %s.\n"
            "Files present: %s" % (
                os.path.dirname(xml_base),
                str(os.listdir(os.path.dirname(xml_base)))))
    facet_xml = candidates[0]
    print("Loading facet MeshFunction from: %s" % facet_xml, flush=True)
    return MeshFunction("size_t", new_mesh, facet_xml)


# -----------------------------------------------------------------------
def rebuild_mesh_bcs(vp_mesh, fsp, rmsh, n_bottom):
    """
    Rebuild ALL mesh BCs on the new mesh/spaces after remesh.
    Called from solve.py immediately after do_remesh().
    """
    # recompute v_normal_proj on new Q_u_dot
    new_v_normal = project(dot(fsp.v_fl_n, n_bottom) * n_bottom, fsp.Q_u_dot)
    vp_mesh.v_normal_proj = new_v_normal

    # u BCs
    vp_mesh.bc_u_0_l = DirichletBC(
        fsp.Q_u.sub(0), Constant(0),
        rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_l_id"])
    vp_mesh.bc_u_0_r = DirichletBC(
        fsp.Q_u.sub(0), Constant(0),
        rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_r_id"])
    vp_mesh.bc_u_t = DirichletBC(
        fsp.Q_u, Constant((0, 0)),
        rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_1_id"])
    vp_mesh.bcs_msh = [vp_mesh.bc_u_0_l, vp_mesh.bc_u_0_r, vp_mesh.bc_u_t]

    # u_dot BCs
    vp_mesh.bc_u_dot_0_l = DirichletBC(
        fsp.Q_u_dot.sub(0), Constant(0),
        rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_l_id"])
    vp_mesh.bc_u_dot_0_r = DirichletBC(
        fsp.Q_u_dot.sub(0), Constant(0),
        rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_r_id"])
    vp_mesh.bc_u_dot_t = DirichletBC(
        fsp.Q_u_dot, Constant((0, 0)),
        rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_1_id"])
    vp_mesh.bc_u_dot_b = DirichletBC(
        fsp.Q_u_dot, vp_mesh.v_normal_proj,
        rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_2_id"])
    vp_mesh.bcs_msh_dot = [
        vp_mesh.bc_u_dot_0_l,
        vp_mesh.bc_u_dot_b,
        vp_mesh.bc_u_dot_0_r,
        vp_mesh.bc_u_dot_t,
    ]


# -----------------------------------------------------------------------
def do_remesh(step, path_to_meshfiles, L, h, gridsize,
              bottom_tag, top_tag, sides_tag):
    """
    Full remesh:
      1. ALE.move bulk mesh by u_n
      2. Extract displaced bottom coords
      3. Build new gmsh mesh
      4. Load new mesh + facet MeshFunction, update sub_meshes[0]
      5. Rebuild function spaces and Function objects on new mesh
      6. Interpolate fluid fields; reset mesh displacement fields to zero
    """
    bulk_mesh = rmsh.lmsh.sub_meshes[0]
    mf_bulk   = rmsh.lmsh.mf_sub_meshes[0]

    # 1. move mesh coordinates by u_n
    ALE.move(bulk_mesh, fsp.u_n)
    print("ALE.move done. ymin=%.6f ymax=%.6f" % (
        bulk_mesh.coordinates()[:, 1].min(),
        bulk_mesh.coordinates()[:, 1].max()), flush=True)

    # 2. extract displaced bottom boundary coordinates
    bottom_coords = _extract_bottom_coords(bulk_mesh, mf_bulk, bottom_tag)
    print("Bottom: x=[%.4f,%.4f] y=[%.4f,%.4f]" % (
        bottom_coords[:, 0].min(), bottom_coords[:, 0].max(),
        bottom_coords[:, 1].min(), bottom_coords[:, 1].max()), flush=True)

    # 3. build new gmsh mesh — returns xml_base (no extension)
    xml_base = build_new_mesh(bottom_coords, step, path_to_meshfiles,
                              L, h, gridsize, bottom_tag, top_tag, sides_tag)
    print("New mesh base: %s" % xml_base, flush=True)

    # 4. save old fluid fields before rebuilding spaces
    old = {
        'v_fl_n_1':      fsp.v_fl_n_1.copy(deepcopy=True),
        'v_fl_n_2':      fsp.v_fl_n_2.copy(deepcopy=True),
        'v_fl_bar':      fsp.v_fl_bar.copy(deepcopy=True),
        'sigma_fl_n_12': fsp.sigma_fl_n_12.copy(deepcopy=True),
        'sigma_fl_n_32': fsp.sigma_fl_n_32.copy(deepcopy=True),
        'phi_fl':        fsp.phi_fl.copy(deepcopy=True),
    }

    # 5. load new mesh and facet MeshFunction, update rmsh module
    new_mesh = Mesh(xml_base + ".xml")
    rmsh.lmsh.sub_meshes[0]    = new_mesh
    new_mf = _load_facet_mf(new_mesh, xml_base)
    rmsh.lmsh.mf_sub_meshes[0] = new_mf

    # rebuild integration measures on new mesh so ds_sub_mesh[0] and
    # dx_sub_mesh[0] are consistent with the new mesh and MeshFunction
    new_dx = Measure("dx", domain=new_mesh)
    new_ds = Measure("ds", domain=new_mesh, subdomain_data=new_mf)
    rmsh.dx_sub_mesh[0] = new_dx
    # rebuild the tagged ds dict — keys must match your mesh read module
    rmsh.ds_sub_mesh[0] = {
        'ds':   new_ds,
        'ds_t': new_ds(rmsh.parameters['sub_mesh_1_id']),
        'ds_b': new_ds(rmsh.parameters['sub_mesh_2_id']),
        'ds_l': new_ds(rmsh.parameters['line_sub_mesh_0_l_id']),
        'ds_r': new_ds(rmsh.parameters['line_sub_mesh_0_r_id']),
        'ds_tb': new_ds(rmsh.parameters['sub_mesh_1_id'])
                + new_ds(rmsh.parameters['sub_mesh_2_id']),
        'ds_lr': new_ds(rmsh.parameters['line_sub_mesh_0_l_id'])
                + new_ds(rmsh.parameters['line_sub_mesh_0_r_id']),
    }

    # rebuild function spaces on new mesh
    fsp.Q_u               = VectorFunctionSpace(new_mesh, 'P', 1)
    fsp.Q_u_dot           = VectorFunctionSpace(new_mesh, 'P', 1)
    fsp.Q_v_fl            = VectorFunctionSpace(new_mesh, 'P', 2)
    fsp.Q_v_fl_bar        = VectorFunctionSpace(new_mesh, 'P', 2)
    fsp.Q_phi_fl          = FunctionSpace(new_mesh, 'P', 1)
    tensor_deg            = fsp.Q_var_tensor_sigma_fl.ufl_element().degree()
    fsp.Q_var_tensor_sigma_fl = TensorFunctionSpace(
        new_mesh, 'P', tensor_deg, shape=(2, 2))

    # rebuild Function objects on new spaces
    fsp.u_n               = Function(fsp.Q_u)
    fsp.u_n_1             = Function(fsp.Q_u)
    fsp.u_n_2             = Function(fsp.Q_u)
    fsp.u_dot_n           = Function(fsp.Q_u_dot)
    fsp.u_dot_n_1         = Function(fsp.Q_u_dot)
    fsp.u_dot_n_2         = Function(fsp.Q_u_dot)
    fsp.v_fl_n            = Function(fsp.Q_v_fl)
    fsp.v_fl_n_1          = Function(fsp.Q_v_fl)
    fsp.v_fl_n_2          = Function(fsp.Q_v_fl)
    fsp.v_fl_bar          = Function(fsp.Q_v_fl_bar)
    fsp.sigma_fl_n_12     = Function(fsp.Q_phi_fl)
    fsp.sigma_fl_n_32     = Function(fsp.Q_phi_fl)
    fsp.phi_fl            = Function(fsp.Q_phi_fl)
    fsp.var_tensor_sigma_fl = Function(fsp.Q_var_tensor_sigma_fl)

    # rebuild Jacobians and test functions
    fsp.J_u               = TrialFunction(fsp.Q_u)
    fsp.J_u_dot           = TrialFunction(fsp.Q_u_dot)
    fsp.nu_u              = TestFunction(fsp.Q_u)
    fsp.nu_u_dot          = TestFunction(fsp.Q_u_dot)
    fsp.J_v_fl_bar        = TrialFunction(fsp.Q_v_fl_bar)
    fsp.J_v_fl_n          = TrialFunction(fsp.Q_v_fl)
    fsp.J_phi_fl          = TrialFunction(fsp.Q_phi_fl)
    fsp.nu_v_fl_n         = TestFunction(fsp.Q_v_fl)
    fsp.nu_v_fl_bar       = TestFunction(fsp.Q_v_fl_bar)
    fsp.nu_phi_fl         = TestFunction(fsp.Q_phi_fl)
    fsp.V_fl              = (fsp.v_fl_n_1 + fsp.v_fl_bar) / 2.0

    # 6. interpolate fluid fields onto new spaces
    parameters['allow_extrapolation'] = True
    fsp.v_fl_n_1.assign(     _interp(old['v_fl_n_1'],      fsp.Q_v_fl))
    fsp.v_fl_n_2.assign(     _interp(old['v_fl_n_2'],      fsp.Q_v_fl))
    fsp.v_fl_bar.assign(     _interp(old['v_fl_bar'],      fsp.Q_v_fl_bar))
    fsp.sigma_fl_n_12.assign(_interp(old['sigma_fl_n_12'], fsp.Q_phi_fl))
    fsp.sigma_fl_n_32.assign(_interp(old['sigma_fl_n_32'], fsp.Q_phi_fl))
    fsp.phi_fl.assign(       _interp(old['phi_fl'],        fsp.Q_phi_fl))
    parameters['allow_extrapolation'] = False

    # mesh displacement fields reset to zero:
    # the new mesh already encodes the deformation — u=0 is the new reference config
    fsp.u_n.vector()[:]       = 0.0
    fsp.u_n_1.vector()[:]     = 0.0
    fsp.u_n_2.vector()[:]     = 0.0
    fsp.u_dot_n.vector()[:]   = 0.0
    fsp.u_dot_n_1.vector()[:] = 0.0
    fsp.u_dot_n_2.vector()[:] = 0.0

    print("Remesh complete. New mesh: %i cells." % new_mesh.num_cells(), flush=True)
    return new_mesh