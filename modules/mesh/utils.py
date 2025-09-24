import colorama as col
import command as cmd
from fenics import *
import numpy as np
import colorama as col
import gmsh
import math
import meshio
import os
import pygmsh

import calculus as cal
import differential_geometry.manifold.geometry as geo
import input_output as io


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh


'''
read the mesh from xdmf file
Input values: 
- 'file name': path and name of the xdmf file
Return values: 
- 'mesh': the mesh
'''


def read_mesh_xdmf(filename):
    mesh = Mesh()

    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mesh)
    xdmf.close()

    return mesh


'''
read the mesh from h5 file
Input values: 
- 'file name': path and name of the h5 file
- 'mesh_name' [optional]: the name of the mesh in the file
Return values: 
- 'mesh': the mesh
'''


def read_mesh_h5(filename, mesh_name='mesh'):
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), filename, "r") as infile:
        infile.read(mesh, mesh_name, False)
    return mesh


'''
Read a mesh from file
Input values: 
- 'file name': path and name of the file, which can be either an xdmf file or h5 file
Return values:
- 'mesh': the mesh
'''


def read_mesh(filename):
    # detect format from file extension
    if filename.endswith('.h5'):
        file_format = "h5"
    elif filename.endswith('.xdmf'):
        file_format = "xdmf"
    else:
        raise ValueError(f"File extension is invalid: {filename}")

    if file_format == "h5":
        return read_mesh_h5(filename)
    elif file_format == "xdmf":
        return read_mesh_xdmf(filename)
    else:
        print(f"File extension is invalid: {filename}")


'''
read the mesh  from  the .msh file 'infile' and write the mesh components (tetrahedra, triangles, lines, vertices) to 'outfile' (tetra_mesh.xdmf, triangle_mesh.xdmf ...)
the component type can be "tetra", "triangle", "line" or "vertex"
if 'prune_z' = true (false), the z component will be removed from the mesh
'''


def write_mesh_components(infile, outfile, component_type, prune_z):
    mesh_from_file = meshio.read(infile)
    # print(f'type of mesh_from_file  = {type(mesh_from_file)}')
    component_mesh = create_mesh(mesh_from_file, component_type, prune_z)
    # print(f'type of component _mesh  = {type(component_mesh)}')
    meshio.write(outfile, component_mesh)


'''
write to .h5 file the components of a mesh determined by a MeshFunction
Input values: 
- 'mesh': the mesh
- 'file_name': the .h5 file where the component will be written
- 'componand_function': the MeshFunction that specifies the component
- 'component_name': the name with which the component will be named in the output file
Example of usage:
    msh.write_mesh_components_h5(mesh_t, io.add_trailing_slash(rarg.args.output_directory) + "line_mesh.h5", cf_t, "cf")
'''


def write_mesh_components_h5(mesh, filename, component_function, component_name):
    with HDF5File(mesh.mpi_comm(), filename, "w") as outfile:
        outfile.write(mesh, "mesh")
        outfile.write(component_function, component_name)


'''
Given a mesh written in an xdmf file, read its components stored into the xdmf file and return the collection of components
Input values: 
- 'mesh': the mesh to read the components from
- 'dim': the dimension of the components to read: example: 1 for lines, 0 for vertices, etc. 
- 'filename': the name of the xdmf file where the components of the mesh are stored
Example: to read the lines of the mesh, call this method with 
    cf = msh.read_mesh_components_xdmf(mesh, 1, args.input_directory + "/line_mesh.xdmf")
'''


def read_mesh_components_xdmf(mesh, dim, filename):
    mesh_value_collection = MeshValueCollection("size_t", mesh, dim)
    with XDMFFile(filename) as infile:
        infile.read(mesh_value_collection, "name_to_read")
        infile.close()
    return cpp.mesh.MeshFunctionSizet(mesh, mesh_value_collection)


'''
Given a mesh written in an h5 file, read its components  stored in an h5 file and returns the collection of components
Input values: 
- 'mesh': the mesh to read the components from
- 'dim': the dimension of the components to read: example: 1 for lines, 0 for vertices, etc. 
- 'filename': the name of the h5 file where the components of the mesh are stored
'''


def read_mesh_components_h5(mesh, dim, filename, name_to_read):
    mesh_function = MeshFunction("size_t", mesh, dim)
    with HDF5File(mesh.mpi_comm(), filename, "r") as infile:
        infile.read(mesh_function, name_to_read)
    return mesh_function


'''
Given a mesh written in a file, read its components stored into the file and return the collection of components
Input values: 
- 'mesh': the mesh to read the components from
- 'dim': the dimension of the components to read: example: 1 for lines, 0 for vertices, etc. 
- 'filename': the name of the file (either .h5 or .xdmf) where the components of the mesh are stored
'''


def read_mesh_components(mesh, dim, filename, name_to_read="name_to_read"):
    # detect format from file extension
    if filename.endswith('.h5'):
        file_format = "h5"
    elif filename.endswith('.xdmf'):
        file_format = "xdmf"
    else:
        raise ValueError(f"File extension is invalid: {filename}")

    if file_format.lower() == "h5":
        '''
        mf = MeshFunction("size_t", mesh, dim)
        with HDF5File(mesh.mpi_comm(), filename, "r") as infile:
            infile.read(mf, mf_name)
        return mf
        '''
        print('Reading mesh components from .h5 file.')
        return read_mesh_components_h5(mesh, dim, filename, name_to_read)


    elif file_format.lower() == "xdmf":
        # mesh_value_collection = MeshValueCollection("size_t", mesh, dim)
        # with XDMFFile(filename) as infile:
        #     infile.read(mesh_value_collection, mf_name)
        #     infile.close()
        # return cpp.mesh.MeshFunctionSizet(mesh, mesh_value_collection)
        print('Reading mesh components from .xdmf file.')

        return read_mesh_components_xdmf(mesh, dim, filename)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")


'''
compare the numerical value of the integral of a test function over a ds, dx, .... with the exact one and output the relative difference and prints out the difference
Input values: 
- 'exact_value': the exact value of the integral
- 'f_test': the function to integrate
- 'meashre': the integration measure
- 'label': the label to be printed out for the integral test
Return values: 
- the absolute value of the relative difference between the finite-element and the exact integral
'''


def test_mesh_integral(exact_value, f_test, measure, label):
    numerical_value = assemble(f_test * measure)

    result = abs((numerical_value - exact_value) / exact_value)
    print(
        f"{label} = {numerical_value:.{4}}, should be {exact_value:.{4}}, relative error =  {result:.{io.number_of_decimals}e}")

    return result


class BoundaryMarker(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


# returns the boundary points of the mesh `mesh`
def boundary_points(mesh):
    # create a dummy function space of degree 1 which will be used only to extract the boundary points
    Q_dummy = FunctionSpace(mesh, 'CG', 1)

    # a map which takes as an input a vertex of Q_dummy.mesh and returns its corresponding degree of freedom
    vertex_to_degree_of_freedom_map = vertex_to_dof_map(Q_dummy)

    # a function which takes as argument the mesh vertices
    vertex_function = MeshFunction("size_t", mesh, 0)

    # set vertex_function -> 1 on the vertices which are part of the boundary (vertex_function is zero elsewhere)
    vertex_function.set_all(0)
    BoundaryMarker().mark(vertex_function, 1)

    # collect the vertices where the vertex_function = 1, i.e., the vertices on the boundary
    boundary_vertices = np.asarray(vertex_function.where_equal(1))

    degrees_of_freedom = vertex_to_degree_of_freedom_map[boundary_vertices]

    x = Q_dummy.tabulate_dof_coordinates()
    x = x[degrees_of_freedom]

    # csvfile = open( "test_boundary_points.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    # print("Degrees of freedom on the boundary:")
    # for degree_of_freedom in degrees_of_freedom:
    # print(f"\t{x[degree_of_freedom]}, {geo.np.linalg.norm( x[degree_of_freedom])}")

    return x


# returns the bulk points of the mesh `mesh`
def bulk_points(mesh):
    # create a dummy function space of degree 1 which will be used only to extract the boundary points
    Q_dummy = FunctionSpace(mesh, 'CG', 1)

    # a map which takes as an input a vertex of Q_dummy.mesh and returns its corresponding degree of freedom
    vertex_to_degree_of_freedom_map = vertex_to_dof_map(Q_dummy)

    # a function which takes as argument the mesh vertices
    vertex_function = MeshFunction("size_t", mesh, 0)

    # set vertex_function -> 1 on the vertices which are part of the boundary (vertex_function is zero elsewhere)
    vertex_function.set_all(0)
    BoundaryMarker().mark(vertex_function, 1)

    # collect the vertices where the vertex_function = 0, i.e., the vertices in the bulk
    boundary_vertices = np.asarray(vertex_function.where_equal(0))

    degrees_of_freedom = vertex_to_degree_of_freedom_map[boundary_vertices]

    x = Q_dummy.tabulate_dof_coordinates()
    x = x[degrees_of_freedom]

    # csvfile = open( "test_bulk_points.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    # print("Degrees of freedom on the boundary:")
    # for degree_of_freedom in degrees_of_freedom:
    # print(f"\t{x[degree_of_freedom]}, {geo.np.linalg.norm( x[degree_of_freedom])}")

    return x


# return the set of boundary points whose distance from the point c lies between r and R
def boundary_points_circle(mesh, r, R, c):
    points = boundary_points(mesh)

    x = []
    for point in points:
        if ((geo.np.linalg.norm(point - c) > r) and (geo.np.linalg.norm(point - c) < R)):
            x.append(point)

    # csvfile = open( "test_boundary_points_circle.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    return x


# compute the lowest and largest x and y values of points in the mesh and return them as a vector in the format [[x_min, x_max], [y_min, y_max]]
def extremal_coordinates(mesh):
    points = boundary_points(mesh)

    if mesh.topology().dim() == 2:

        x_min = points[0][0]
        x_max = x_min
        y_min = points[0][1]
        y_max = y_min

        for point in points:
            if point[0] < x_min:
                x_min = point[0]

            if point[0] > x_max:
                x_max = point[0]

            if point[1] < y_min:
                y_min = point[1]

            if point[1] > y_max:
                y_max = point[1]

        # print(f"\textremal coordinates: {x_min}, {x_max}, {y_min}, {y_max}")

        return [[x_min, x_max], [y_min, y_max]]

    elif mesh.topology().dim() == 1:

        x_min = points[0][0]
        x_max = x_min

        for point in points:
            if point[0] < x_min:
                x_min = point[0]

            if point[0] > x_max:
                x_max = point[0]


        print(f"\textremal coordinates: {x_min}, {x_max}")

        return [x_min, x_max]
'''
compute the difference between functions f and g on the boundary of the mesh on which f and g are defined, returning 
sqrt(\sum_{i \in {vertices in the boundary of the mesh} [f(x_i) - g(x_i)]^2/ (number of vertices in the boundary of the mesh})
'''


def difference_on_boundary(f, g):
    mesh = f.function_space().mesh()
    boundary_points_mesh = boundary_points(mesh)

    # print("\n\nx\tf(x)-g(x)")
    diff = 0.0
    for x in boundary_points_mesh:
        delta = f(x) - g(x)
        diff += (delta ** 2)

    diff = np.sqrt(diff / len(boundary_points_mesh))

    return diff


'''
compute the difference between functions f and g in the bulk of the mesh on which f and g are defined, returning 
sqrt(\sum_{i \in {vertices in the bulk of the mesh} [f(x_i) - g(x_i)]^2/ (number of vertices in the bulk of the mesh})
'''


def difference_in_bulk(f, g):
    mesh = f.function_space().mesh()
    bulk_points_mesh = bulk_points(mesh)

    diff = 0.0
    for x in bulk_points_mesh:
        delta = f(x) - g(x)
        diff += (delta ** 2)

    diff = np.sqrt(diff / len(bulk_points_mesh))

    return diff


# return sqrt(<(f-g)^2>_measure / <measure>), where measure can be dx, ds_...
def difference_wrt_measure(f, g, measure):
    return sqrt(assemble(((f - g) ** 2 * measure)) / assemble(Constant(1.0) * measure))


# return sqrt(<f^2>_measure / <measure>), where measure can be dx, ds_...
def abs_wrt_measure(f, measure):
    return difference_wrt_measure(f, Constant(0), measure)


'''
compute the difference between functions f and g on the boundary of the mesh, boundary_c, given by the boundary points whose distance from point c lies between r and R, returning 
sqrt(\sum_{i \in {vertices in boundary_c} [f(x_i) - g(x_i)]^2/ (number of vertices in boundary_c})
'''


def difference_on_boundary_circle(f, g, r, R, c):
    mesh = f.function_space().mesh()
    boundary_c_points = boundary_points_circle(mesh, r, R, c)

    diff = 0.0
    for x in boundary_c_points:
        delta = f(x) - g(x)
        diff += (delta ** 2)

    diff = np.sqrt(diff / len(boundary_c_points))

    return diff


'''
write to csv file 'outfile' the coordinates of the start and end vertices which define the lines of the triangles of a 2d mesh stored in the .msh file 'infile'
the vertices are written in the format
edge1_start[0], edge1_start[1], edge1_start[2], edge1_end[0], edge1_end[1], edge1_end[2]
edge2_start[0], edge2_start[1], edge2_start[2], edge2_end[0], edge2_end[1], edge2_end[2]
...
'''


def print_mesh_lines_to_csv(infile, outfile):
    # open the .msh file
    gmsh.open(infile)

    # get the list of components with dimension 2 from the mesh (triangles)
    triangles = gmsh.model.mesh.getElements(dim=2)
    # print( "triangles = ", triangles )

    # construct a map which, given the tag of a node, gives its coordinates
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_map = {node_tags[i]: node_coords[3 * i: 3 * (i + 1)] for i in range(len(node_tags))}
    # print( "node map = ", node_map )

    # Store unique edges from the triangle elements
    # initialize a 'list' of unique elements, this sets the list to empty
    edges = set()

    # loop over all triangle nodes
    triangle_nodes = triangles[2][0] if len(triangles[2]) > 0 else []
    for i in range(0, len(triangle_nodes), 3):
        # store into pair_12 = [ID_1, ID_2] the IDs of the vertices which lie at the extremities of the line in the triangle, and similarly for pair_23, pair_31
        pair_12 = tuple(sorted([triangle_nodes[i], triangle_nodes[i + 1]]))
        pair_23 = tuple(sorted([triangle_nodes[i + 1], triangle_nodes[i + 2]]))
        pair_31 = tuple(sorted([triangle_nodes[i + 2], triangle_nodes[i]]))

        # this pushes back the elements pair_12, pair_23, pair_31 to edges
        edges.update([pair_12, pair_23, pair_31])
        # print( f"pair_12 = {pair_12} pair_23 = {pair_23} pair_31 = {pair_31}" )

    # loop through the edges added before and write the endoints of their lines to file
    csvfile = open(outfile, "w")
    print(f"\"start:0\",\"start:1\",\"start:2\",\"end:0\",\"end:1\",\"end:2\"", file=csvfile)
    for edge in edges:
        # apply node_map to obtain the coordinates of the starting vertex in edge from their IDs, and similarly for p_end
        p_start = node_map[edge[0]]
        p_end = node_map[edge[1]]
        # print( f"\tEdge from {edge[0]} to {edge[1]}: p_start = ({p_start[0]}, {p_start[1]}, {p_start[2]}), "p_end = ({p_end[0]}, {p_end[1]}, {p_end[2]})" )
        print(f"{p_start[0]}, {p_start[1]}, {p_start[2]},{p_end[0]}, {p_end[1]}, {p_end[2]}", file=csvfile)

    csvfile.close()


'''
print the coordinates of start and end points of line 'line'
'''


def print_line_info(line, label):
    # Get the start and end points of the specific line
    start_point, end_point = get_line_extrema(line)

    print(f"\t{label}:\n\t\ttag = {line}")
    print_point_info(start_point, 'start_point')
    print_point_info(end_point, 'end_point')


# print the coordiantes of point 'point'
def print_point_info(point, label):
    r = get_point_coordinates(point)
    print(f"\t{label}:\n\t\ttag = {point},\n\t\tcoordinates =  {r}")

    return r


# print the info of all points in list 'list', which has label 'label'
def print_point_list_info(list, label):
    print(f'{label}: length = {len(list)}\ncontent:')
    for i in range(len(list)):
        print_point_info(list[i], f'point #{i}')


'''
add a line given by n-1 segments  separated by n points, between a point and a coordinate
- 'p_start' : ID of the starting point of the line
- 'r_end' :  coordinate of end point of the line
- 'n': number of points

Returns 
- 'points': a list of IDs of the points added as part of the line
- 'segments': a list of IDs of segments added as part of the line 
'''


def add_line_p_start_r_end_n(p_start, r_end, n, model):
    # print("Generating line ... ")

    points = [p_start]
    segments = []

    # coordinates of the start point
    r_start = get_point_coordinates(p_start)

    if n > 1:

        for i in range(1, n):
            dr = np.subtract(r_end, r_start)
            dr *= i / (n - 1)
            points.append(add_point(np.add(r_start, dr), model))

            segments.append((add_line_p_start_p_end(points[i - 1], points[i], model))[1])

            # print_point_info(points[-1], 'last added point')
            # print_line_info(segments[-1], 'last added segment')

        # print("... done.")

    else:
        print("Cannot add points!! ")

    return points, segments


'''
add a line given by n-1 segments  separated by n points, between two points
- 'p_start' : ID of the starting point of the line
- 'r_end' :  coordinate of end point of the line
- 'n': number of points

Returns 
- 'points': a list of IDs of the points added as part of the line
- 'segments': a list of IDs of segments added as part of the line 
'''


def add_line_p_start_p_end_n(p_start, p_end, n, model):
    # print("Generating line ... ")

    points = [p_start]
    segments = []

    # coordinates of the start point
    r_start = get_point_coordinates(p_start)
    r_end = get_point_coordinates(p_end)

    if n > 1:

        for i in range(1, n - 1):
            dr = np.subtract(r_end, r_start)
            dr *= i / (n - 1)
            points.append(add_point(np.add(r_start, dr), model))

            segments.append(add_line_p_start_p_end(points[i - 1], points[i], model)[1])

            # print_point_info(points[-1], 'last added point')
            # print_line_info(segments[-1], 'last added segment')

        # print("... done.")

        points.append(p_end)
        model.synchronize()

        segments.append(add_line_p_start_p_end(points[n - 2], p_end, model)[1])

        # print_point_info(points[-1], 'last added point')
        # print_line_info(segments[-1], 'last added segment')

    else:
        print("Cannot add points!! ")

    return points, segments


'''
add point with coordinates 'r' to model 'model' and return the result
'''


def add_point(r, model):
    point = model.add_point(r[0], r[1], r[2])
    model.synchronize()

    return point


'''
add a line between points 'p_start' and 'p_end' in model 'model' and return the line
'''


def add_line_p_start_p_end(p_start, p_end, model):
    line = model.add_line(p_start, p_end)
    model.synchronize()

    return [p_start, p_end], line


'''
add a line betweeen point 'p_start' and a new point with coordiantes r_end, which will be created, and return the line 

'''


def add_line_p_start_r_end(p_start, r_end, model):
    p_end = add_point(r_end, model)
    points_start_end, line = add_line_p_start_p_end(p_start, p_end, model)

    return points_start_end, line


'''
add a line between two points by setting the point coordinates
- 'r_start' : coordinates of the start point
- 'r_end' : coordinates of the end point
- 'model' : meshing model

return values:
- a list with the start and end point
- the line
'''


def add_line_r_start_r_end(r_start, r_end, model):
    p_start = add_point(r_start, model)
    p_end = add_point(r_end, model)
    points_start_end, line = add_line_p_start_p_end(p_start, p_end, model)

    return points_start_end, line


# get the coordinates of the vertex 'vertex', where vertex[0] is the dimension of the vertex (0) an vertex[1] the vertex tag (id)
def get_point_coordinates(point):
    return gmsh.model.getValue(0, point, [])  # 0 = vertex dimension


'''
return extermal points of line 'line'
'''


def get_line_extrema(line):
    start_point, end_point = gmsh.model.getAdjacencies(1, line)[1]  # [1] gives point tags

    return start_point, end_point


'''
return the coordinates of the center of mass of line 'line'
'''


def get_line_center_of_mass_coordinates(line):
    start_point, end_point = get_line_extrema(line)

    start_r = get_point_coordinates(start_point)
    end_r = get_point_coordinates(end_point)

    return (np.add(start_r, end_r) / 2)


'''
sort a list of vertices
- 'vertex_list': a list of vertices: vertex_list[i] = [ vertex_dimension (=0), vertex_id ]
- 'direction_id': the ID of the coordinate according to which the list will be sorted: 
    * to sort according to the x coordinate set direction_id = 0, 
    * to sort according to the y coordinate set direction_id = 1, 
    * to sort according to the z coordinate set direction_id = 2, 
- 'reverse': if True, the list will be sorted with respect to increasing order of the coordinate 'coordinate_id', and in reverse order otherwise
Return values:
- the sorted list of vertices
'''


def sort_vertex_list(vertex_list, direction_id, reverse):
    point_coordinates = []

    for vertex in vertex_list:
        coordinates = get_point_coordinates(vertex[1])
        point_coordinates.append([vertex, coordinates])

    point_coordinates.sort(key=lambda x: x[1][direction_id], reverse=reverse)
    print(f'sorted list = {point_coordinates}')

    return point_coordinates


'''
create a circle composed of four arcs
- 'c_r' : coordinates of the center of the circle
- 'r' : circle radius
- 'model' the meshing model used

return values:
- the circle lines (the four arcs)
- the circle points
'''


def add_circle_with_arcs(c_r, r, model):
    # add the center of the circle
    p_c = add_point(c_r, gmsh.model.geo)

    # add the point on the left, 'p_l', on the right 'p_r', on the top 'p_t' and on the bottom 'p_b'
    p_l = add_point(np.subtract(c_r, [r, 0, 0]), model)
    p_r = add_point(np.add(c_r, [r, 0, 0]), model)
    p_t = add_point(np.add(c_r, [0, r, 0]), model)
    p_b = add_point(np.subtract(c_r, [0, r, 0]), model)

    # add four arcs which will make the circle: add the arc from p_r to p_t , and similarly for the other arcs
    arc_rt = model.add_circle_arc(p_r, p_c, p_t)
    model.synchronize()

    arc_tl = model.add_circle_arc(p_t, p_c, p_l)
    model.synchronize()

    arc_lb = model.add_circle_arc(p_l, p_c, p_b)
    model.synchronize()

    arc_br = model.add_circle_arc(p_b, p_c, p_r)
    model.synchronize()

    circle_lines = [arc_rt, arc_tl, arc_lb, arc_br]

    # add the circle loop
    circle_loop = model.add_curve_loop(circle_lines)
    model.synchronize()

    return circle_lines, circle_loop


'''
create a circle composed of multiple segments
- 'c_r' : coordinates of the center of the circle
- 'r' : circle radius
- 'n_segments': the number of segments
- 'model' the meshing model used

return values:
- the circle points
- the circle segments
'''


def add_circle_with_lines(c_r, r, n_segments, model):
    points_circle = []
    segments_circle = []

    coord = np.add(c_r, [r, 0, 0])
    points_circle.append(add_point(coord, model))

    for i in range(1, n_segments - 1):
        coord = np.add(c_r, np.dot(cal.R_z(i / (n_segments - 1) * 2.0 * np.pi), [r, 0, 0]))
        points_circle.append(add_point(coord, model))
        segments_circle.append((add_line_p_start_p_end(points_circle[i - 1], points_circle[i], model))[1])

    segments_circle.append(add_line_p_start_p_end(points_circle[-1], points_circle[0], model)[1])

    return points_circle, segments_circle


'''
tag as physical entities the objects with a given dimension in a mesh
Input values:
- 'list_of_objects': an array containing the objects to be tagged
- 'dimension': the dimension of the objects that one wants to tag
- 'tag' : the tag which one wants to give to the objects
- 'labal' : the lable which one wants to give to the objects
'''


def tag_group(list_of_objects, dimension, tag, label):
    gmsh.model.addPhysicalGroup(dimension, list_of_objects, tag)
    gmsh.model.setPhysicalName(dimension, tag, label)


'''
Print the information on a triangle in a mesh
Input values:
- 'triangle': the triangle, an element of mesh.cells[i].data
- 'mesh': the mesh
'''


def print_mesh_triangle(triangle, mesh):
    # vertex_1 = tuple(sorted([triangle[0], triangle[1]]))
    # vertex_2 = tuple(sorted([triangle[1], triangle[2]]))
    # vertex_3 = tuple(sorted([triangle[2], triangle[0]]))
    coordinates_vertex_1 = mesh.points[triangle[0]]
    coordinates_vertex_2 = mesh.points[triangle[1]]
    coordinates_vertex_3 = mesh.points[triangle[2]]

    print(f'\tTriangle {np.sort(triangle)}')
    print(f'\t\t{coordinates_vertex_1}\n\t\t{coordinates_vertex_2}\n\t\t{coordinates_vertex_3}')


'''
Print all triangles of a mesh
Input values 
- 'mesh': the mesh, a <meshio mesh object>
'''


def print_mesh_triangles(mesh):
    print('Cell triangles: ')
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            for triangle in cell_block.data:
                print_mesh_triangle(triangle, mesh)


'''
Print all mesh vertices
Input values: 
- 'mesh': the mesh, a <meshio mesh object>
'''


def print_mesh_vertices(mesh):
    for i, point in enumerate(mesh.points):
        print(f"Vertex ID: {i}, Coordinates: {point}")


'''
Print all element types of a mesh (such as triangles, tetrahedra, lines ...)
Input values: 
- 'mesh': the mesh, a <meshio mesh object>
'''


def print_mesh_element_types(mesh):
    print("Cell types in the mesh:")
    for cell_block in mesh.cells:
        print(f"\t{cell_block.type}")


'''
Print the lines of a mesh
Input values 
- 'mesh': the mesh, a <meshio mesh object>
'''


def print_mesh_lines(mesh):
    print('Cell lines: ')

    for j in range(len(mesh.cells)):
        # loop through  blocks of lines

        if mesh.cells[j].type == "line":
            print(f'\tLine block {mesh.cells[j].data}')

            # loop through the lines in  block  mesh.cells[j].data
            for i in range(len(mesh.cells[j].data)):
                # obtain the extremal point of each line
                vertex_1 = mesh.points[mesh.cells[j].data[i][0]]
                vertex_2 = mesh.points[mesh.cells[j].data[i][1]]

                print(f"\t\tLine: {i}:\n\t\t\t{vertex_1}\n\t\t\t{vertex_2}")


'''
print information (element types, triangles, vertices) on a mesh
Input values: 
- 'mesh': the mesh, a <meshio mesh object>
- 'title' : a title for the printout
'''


def print_mesh_info(mesh, title):
    print(f'{title}')
    print_mesh_element_types(mesh)
    print_mesh_triangles(mesh)
    print_mesh_vertices(mesh)


'''
assign a tag to lines in a cell which satisfy a given condition
Input values:
- 'line_condition': a function of the line which tells whether the line satifies the condition to be tagged
- 'tag' : the tag which one wants to assign to the lines
- 'mesh': the mesh, a <meshio mesh object>
'''


def asssign_tag_to_lines(line_condition, tag, mesh):
    # assign to the l edge the id 'lower_edge_id'
    for j in range(len(mesh.cells)):
        # loop through  blocks of lines

        if mesh.cells[j].type == "line":
            # print(f'\tI am on line block {mesh.cells[j].data}')

            # loop through the lines in  block  mesh.cells[j].data
            for i in range(len(mesh.cells[j].data)):

                if line_condition(mesh.cells[j].data[i]):
                    # the extremal points lie on the axis x[1] = 0 -> the line mesh.cells[j].data[i] belongs to the b edge of the rectangle
                    # print(f"\t\tLine: {i} -> Point 1: {point1}, Point 2: {point2}")
                    # tag the line under consideration with ID target_id
                    mesh.cell_data['gmsh:physical'][j][i] = tag


'''
This function mirrors the points in a rectangular mesh: 
Input values: 
- 'mirror_function': the function which performs the mirroring of each point
- 'points' : Array of points to be duplicated
- 'point_data' : Data that contains dimensional tag of the points (must be duplicated as well to avoid issues during the reading of the mesh)
Return values: 
- 'new_points' : the old and the new points
- 'non_mirrored_new_points_indices' : the indices of the old points which have not been mirrored, and of the 
newly mirrored points in the new array 
(they are not just the indices of the old points traslated by some constant since the points on the x axis has not been duplicated and they were not ordered in the old list)
- 'mirrored_point_data ': array of the points which have been mirrored 

Example of usage: 
'''


def mirror_points(axis_of_symmetry_condition, mirror_function, points, point_data):
    offset = 0
    non_mirrored_plus_new_points_indices = []
    mirrored_points = []
    mirrored_point_data = []

    print('Called mirror_points. Looping through points to mirror them ...')

    for i in range(len(points)):
        # if np.isclose(points[i, 1], axis_of_symmetry_condition, rtol=cal.small_number):
        if axis_of_symmetry_condition(points[i]):
            # I ran into a point with x[1] = y_coordinate_axis_of_symmetry -> do not mirror it and append to old_plus_new_points the same index 'i' as the original point
            offset += 1
            non_mirrored_plus_new_points_indices.append(i)

            # print(f'\tNot mirroring points with label {i}')

        else:
            #  I ran into a point with x[1] != y_coordinate_axis_of_symmetry -> mirror it
            non_mirrored_plus_new_points_indices.append(i - offset + len(points))
            l = list(point_data['gmsh:dim_tags'][i, :])

            # append two points with indexes:
            # 1) the original point
            mirrored_point_data.append(l)
            # 2) the mirror of the original point
            # mirrored_points.append([points[i, 0], h - points[i, 1], points[i, 2]])
            mirrored_points.append(mirror_function(points[i]))

    print('... done.')

    mirrored_points = np.array(mirrored_points)
    old_plus_new_points = np.vstack((points, mirrored_points))

    return old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data


'''
mirrors lines in a mesh according to an axis of symmetry
Input values:
- 'mesh': the mesh, a <meshio mesh object>
- 'gamma_axis_of_symmetry': the curve which defines the axis of symmetry
- 'non_mirrored_plus_new_points_indices': the indices of the old points which have not been mirrored, and of the new points, as returned from 'mirror_points'

Example of usage: 
old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, mirror_function, mesh.points,
 msh.mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)                                                                                                  mesh.point_data)
'''


def mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices):
    print('Duplicating cell lines ... ')

    for j in range(len(mesh.cells)):
        # print(f'\tj = {j}', flush=True)

        if mesh.cells[j].type == 'line':
            lines = np.copy(mesh.cells[j].data)
            filtered_lines = []

            # print(f'\t\tlines = {lines}')

            for i in range(np.shape(lines)[0]):

                # print(f'\t\t\tlines[i] = {lines[i]}')

                if (not cal.line_on_axis(lines[i], gamma_axis_of_symmetry, mesh)):
                    filtered_lines.append([non_mirrored_plus_new_points_indices[lines[i, 0]],
                                           non_mirrored_plus_new_points_indices[lines[i, 1]]])

                    # print('\t\t\t\tLine has been mirrored')

                # else:
                # print('\t\t\t\tLine has not been mirrored')

            filtered_lines = np.array(filtered_lines)

            # print(f'\t\tfiltered_lines = {filtered_lines}', flush=True)

            if filtered_lines != []:
                lines_plus_filtered_lines = np.vstack((lines, filtered_lines))
            else:
                lines_plus_filtered_lines = lines

            # print(f'\t\tlines + filetered lines = {lines_plus_filtered_lines}', flush=True)

            mesh.cells[j] = meshio.CellBlock("line", lines_plus_filtered_lines)

            N = np.shape(mesh.cells[j].data)[0]

            # print(f'\t\tN = {N}', flush=True)
            # print(f'\t\tcell_data["gmsh:physical"][{j}] = {mesh.cell_data["gmsh:physical"][j]}', flush=True)

            mesh.cell_data['gmsh:physical'][j] = np.array([mesh.cell_data['gmsh:physical'][j][0]] * N)
            mesh.cell_data['gmsh:geometrical'][j] = np.array([mesh.cell_data['gmsh:geometrical'][j][0]] * N)

    print('... done.')


'''
mirror the triangles in a cell
- 'mesh': the mesh, a <meshio mesh object>
- 'old_plus_new_points' : the set of old and new (mirrored) points, as returned from 'mirror_points'
- 'non_mirrored_plus_new_points_indices': the indices of the non-mirrored and new points, as returned from 'mirror_points'
- 'mirrored_point_data': data of the mirrored poitns, as returned from 'mirror_points'

'''


def mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data):
    old_triangles = mesh.cells_dict['triangle']

    # duplicate cell blocks of type 'triangle'
    new_triangles = np.copy(old_triangles)

    # run through the old triangles
    for i in range(np.shape(new_triangles)[0]):
        # for each old triangle, run through each of its three vertices
        for j in range(3):
            '''
            assign to the new triangle the vertex tag of the old triangle, mapped towards the vertex tags of the mirrored vertices
            In this way, one reconstructs the same pattern as the old triangles, for the flipped part of the mesh
            '''
            new_triangles[i, j] = non_mirrored_plus_new_points_indices[old_triangles[i, j]]

    mesh.points = old_plus_new_points
    mesh.point_data['gmsh:dim_tags'] = np.vstack((mesh.point_data['gmsh:dim_tags'], mirrored_point_data))
    mesh.cells[-1] = meshio.CellBlock("triangle", np.vstack((old_triangles, new_triangles)))
    N = np.shape(mesh.cells[-1].data)[0]
    mesh.cell_data['gmsh:physical'][-1] = np.array([mesh.cell_data['gmsh:physical'][-1][0]] * N)
    mesh.cell_data['gmsh:geometrical'][-1] = np.array([mesh.cell_data['gmsh:geometrical'][-1][0]] * N)


'''
mirror a mesh with respect to an axis of symmetry
Input values: 
- 'mesh': the mesh, a <meshio mesh object>
- 'gamma_axis_of_symmetry': the curve which defines the axis of symmetry

Example of usage:
gamma_axis_of_symmetry = lambda t: cal.line(r_1, r_4, t)
msh.mirror_mesh(mesh, gamma_axis_of_symmetry)
'''


def mirror_mesh(mesh, gamma_axis_of_symmetry):
    # define the function which tells whether a point is on the axis of symmetry
    f_on_axis_of_symmetry = lambda point: cal.point_on_line(point, gamma_axis_of_symmetry)

    # define the function which mirrors the coordinates of a point with respect to the axis of symmetry
    f_mirror = lambda point: cal.mirror_point_line(point, gamma_axis_of_symmetry)

    # mirror  mesh points and return the relative data
    old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = mirror_points(f_on_axis_of_symmetry, f_mirror, mesh.points,
                                                                                                   mesh.point_data)
    # mirror  mesh triangles
    mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data)

    # mirror mesh lines
    mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)


'''
check the l <-> symmetry of a square mesh
Input values :
- 'mesh': the mesh, a <meshio mesh object>
- 'center': the center with respect to which symmetry will be assessed. This method will assess the symmetry with respect to the lines
  x[0] = center[0] (line parallel to the x[1] axis) and with respect to the line x[1] = center[1] (line parallel to the x[0] axis)

Example of usage:
    msh.check_lr_symmetry_square_mesh(mesh, c)
'''


def check_mesh_symmetry(mesh, center):
    Q = FunctionSpace(mesh, 'CG', 1)
    coordinates = Q.tabulate_dof_coordinates()

    print(f'Number of vertices = {Q.dim()}')

    average_lr = 0
    n_vertices_average_lr = 0

    average_tb = 0
    n_vertices_average_tb = 0

    for i in range(Q.dim()):

        if ((not np.isclose(coordinates[i][0], center[0]))):
            average_lr += coordinates[i][0]
            n_vertices_average_lr += 1

        if ((not np.isclose(coordinates[i][1], center[1]))):
            average_tb += coordinates[i][1]
            n_vertices_average_tb += 1

    average_lr /= n_vertices_average_lr
    average_tb /= n_vertices_average_tb

    print(f'Check l <-> r symmetry: <x - center_x> = {col.Fore.BLUE}{(average_lr - center[0]):.{io.number_of_decimals}e}{col.Fore.RESET}')
    print(f'Check t <-> b symmetry: <y - center_y> = {col.Fore.BLUE}{(average_tb - center[1]):.{io.number_of_decimals}e}{col.Fore.RESET}')


'''
Generate a mesh given by a ring slice
Input values: 
- 'r', 'R': the inner and outer radii of the circles delimiting the ring
- 'c_r', 'c_R' the centers of the rings
- 'theta': the angular width of the slice, in radians
- 'resolution': the mesh resolution
- 'output_file': the .msh file where the mesh will be stored. The mesh lines will be written in the same folder in line_vertices.csv file

Example of usage:
    msh.generate_mesh_ring_slice(r, R, c_r, c_R, theta, resolution, mesh_slice_file)
'''


def generate_mesh_ring_slice(r, R, c_r, c_R, theta, resolution, output_file):
    output_directory = io.add_trailing_slash(os.path.dirname(output_file))

    # create the path for the csv file if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    surface_id = 1
    circle_r_id = 2
    circle_R_id = 3
    line_t_id = 4
    line_b_id = 5
    ids = [1, line_b_id, circle_R_id, circle_r_id, line_t_id]

    #  mesh is generated used pygmsh and it's saved in slice_mesh_msh_file
    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()

    print(f'r = {r}\nr = {R}\nc_r = {c_r}\nc_R = {c_R}\nresolution = {resolution}\noutput directory = {output_file}')

    # center points, used to define the arcs
    p_c_r = model.add_point((c_r[0], c_r[1], 0))
    p_c_R = model.add_point((c_R[0], c_R[1], 0))

    # extremal points of the ring slice
    r_1 = np.array([r, 0])
    r_2 = cal.R(theta).dot(r_1)
    r_4 = np.array([R, 0])
    r_3 = cal.R(theta).dot(r_4)

    p_1 = model.add_point((r_1[0], r_1[1], 0), mesh_size=resolution)
    p_2 = model.add_point((r_2[0], r_2[1], 0), mesh_size=resolution)
    p_3 = model.add_point((r_3[0], r_3[1], 0), mesh_size=resolution)
    p_4 = model.add_point((r_4[0], r_4[1], 0), mesh_size=resolution)
    model.synchronize()

    arc_12 = model.add_circle_arc(p_1, p_c_r, p_2)
    model.synchronize()

    line_23 = model.add_line(p_2, p_3)
    model.synchronize()

    arc_34 = model.add_circle_arc(p_3, p_c_r, p_4)
    model.synchronize()

    line_41 = model.add_line(p_4, p_1)
    model.synchronize()

    slice_lines = [arc_12, line_23, arc_34, line_41]
    slice_loop = model.add_curve_loop(slice_lines)
    model.synchronize()

    slice_surface = model.add_plane_surface(slice_loop)
    model.synchronize()

    model.add_physical([slice_surface], "Volume")
    model.add_physical([slice_lines[0]], "r")
    model.add_physical([slice_lines[2]], "R")
    model.add_physical([slice_lines[1]], "top")
    model.add_physical(slice_lines[3], "bottom")

    geometry.generate_mesh(dim=2)
    gmsh.write(output_file)

    print_mesh_lines_to_csv(output_file, output_directory + 'line_vertices.csv')

    gmsh.clear()
    geometry.__exit__()


"""
Translates the coordinates of each point in the mesh by the displacement field u.
This function returns a new mesh with the translated coordinates.

Parameters:
- 'mesh': the original Mesh
- 'u': the displacement field, a Function in a VectorFunctionSpace defined over the mesh

Returns:
- a Mesh object with deformed coordinates, and same ids and mesh structure
"""


def deform_mesh(mesh, u):
    # Copy the mesh to avoid modifying the original
    deformed_mesh = Mesh(mesh)

    # Create a coordinate map for modifying vertex coordinates
    new_mesh_coordinates = deformed_mesh.coordinates()
    mesh_dimension = mesh.geometry().dim()

    # Loop over all vertex coordinates and apply displacement
    for i in range(len(new_mesh_coordinates)):
        new_mesh_coordinate = new_mesh_coordinates[i]
        value_u = u(new_mesh_coordinates[i])  # Evaluate displacement at this point
        new_mesh_coordinates[i] = new_mesh_coordinate + value_u

    return deformed_mesh


'''
full write of mesh data to file
Input values: 
- 'mesh_file': the .msh file where the mesh is stored
- 'components': a list of the components of the mesh to be written, e.g., ['tetra', 'triangle', 'line', 'vertex']. 
    They must be inserted in decreasing order of dimension of the component: for example 'triangle' before 'vertex'
- 'parameters': a dictionary of mesh parameters
- 'output_directory': the path where the mesh info will be written
- 'prune_z': whether the z component should be pruned (true) or not (false)

Example of usage:
    msh.full_write(mesh_file, ['triangle', 'line', 'vertex'], rpam.parameters, output_directory, True)
'''


def full_write(mesh_file, components, parameters, output_directory, prune_z):
    output_directory_slash = io.add_trailing_slash(output_directory)

    for component in components:
        write_mesh_components(mesh_file, output_directory_slash + component + "_mesh.xdmf", component, prune_z)

    # print  mesh vertices to csv file
    mesh = read_mesh(output_directory_slash + components[0] + "_mesh.xdmf")
    io.print_mesh_vertices_to_csv(mesh, output_directory_slash + "vertices.csv")

    # print the mesh lines to csv fie
    print_mesh_lines_to_csv(mesh_file, output_directory_slash + "line_vertices.csv")

    # print mesh metadata
    io.write_parameters_to_csv_file(output_directory_slash + "mesh_metadata.csv", parameters)


'''
Given a parent mesh and a submesh of it, and function mf_parent which identifies facets on the parent mesh, 
this method returns the function which identifies the facet markers on the  sub_mesh, with the same ids as in the parent mesh
Input values: 
- 'parent': the parent mesh
- 'submesh': the submesh of the parent mesh
- 'mf_parent': the function which identifies facets on the parent mesh
Return values
- 'mf_submesh': the function which identifies facets on a submesh of the parent mesh

Example of usage: 
    mf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")
    submesh_out = SubMesh(lmsh.mesh, sf, parameters["surface_out_id"])
    mf_submesh_out = transfer_facet_tags_to_sub_mesh(lmsh.mesh, submesh_out, mf)
    
Then you can create a ds on the submesh with 
    ds_l_submesh_out = Measure("ds", domain=submesh_out, subdomain_data=mf_submesh_out, subdomain_id=parameters["line_sub_mesh_1_l_id"])
'''


def transfer_facet_tags_to_sub_mesh(parent_mesh, sub_mesh, mf_parent):
    # Create facet marker on submesh
    mf_sub = MeshFunction('size_t', sub_mesh, 1, 0)

    vertex_map = sub_mesh.data().array("parent_vertex_indices", 0)

    # run through all the facets of the sub_mesh
    for sub_mesh_facet in facets(sub_mesh):
        # extract the vertices of the facet under considerationn
        sub_mesh_facet_vertices = sub_mesh_facet.entities(0)

        # consider the relative vertices in the parent mesh
        parent_vertices = [vertex_map[v] for v in sub_mesh_facet_vertices]
        #  search for a facet in the parent mesh that shares these
        for facet in facets(parent_mesh):
            if sorted(facet.entities(0)) == sorted(parent_vertices):
                # a corresponding facet in the parent mehs has been found
                mf_sub[sub_mesh_facet.index()] = mf_parent[facet.index()]
                break

    return mf_sub


'''
map the tags of  boundary lines of a parent mesh to a boudnary mesh derived from the parent mesh
Input values: 
- 'boundary_mesh': the boundary mesh obtained from the parent mesh
- 'mf_parent_mesh' : the map which tags the lines in the parent mesh
Return values: 
- 'mf_boundary_mesh': the map which tags the lines in the boundary mesh, with the same ids which they had in the parent mesh

Example of usage: 
    submesh_out = SubMesh(parent_mesh, sf, rpam.parameters["surface_out_id"])
    boundary_mesh = BoundaryMesh(submesh_out, "exterior", order=True)
    mf_submesh_out = msh.transfer_facet_tags_to_sub_mesh(parent_mesh, submesh_out, mf)
    mf_boundary_mesh = msh.transfer_facet_tags_to_bounday_mesh(boundary_mesh, mf_submesh_out)
'''


def transfer_facet_tags_to_bounday_mesh(boundary_mesh, mf_parent_mesh):
    # entity_map(1) maps boundary mesh facets to sub_mesh facets
    boundary_to_parent_facet_map = boundary_mesh.entity_map(1)

    # construct a map function which tags all vertices (dimension = 1), with id 0
    mf_boundary_mesh = MeshFunction("size_t", boundary_mesh, 1, 0)  # facets in 1D mesh are edges

    # run on all facets of boundary_mesh
    for i, b_facet in enumerate(facets(boundary_mesh)):
        # obtain the id with whuch the facet under consideration  was tagged in boundary mesh
        submesh_facet_id = boundary_to_parent_facet_map[i]
        # impose that the function  mf_boundary_mesh evaluated on the facet under consideration must be equal to the id that the facet had in the submesh
        mf_boundary_mesh[b_facet] = mf_parent_mesh[submesh_facet_id]

    return mf_boundary_mesh


'''
Given a parent mesh and a submesh of it, and function sf_parent which identifies cells on the parent mesh, 
this method returns the function which identifies the cells on the sub_mesh, with the same ids as in the parent mesh
Input values: 
- 'sub_mesh': the sub_mesh of the parent mesh
- 'sf_parent': the function which identifies cells on the parent mesh
Return values
- 'sf_submesh': the function which identifies cells on the sub_mesh of the parent mesh

Example of usage: 
    sf = msh.read_mesh_components(lmsh.mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
    submesh_out = SubMesh(lmsh.mesh, sf, parameters["surface_out_id"])
    sf_submesh_out = msh.transfer_cell_tags_to_sub_mesh(submesh_out, sf)

Then you can create a ds on the submesh with 
 

'''


def transfer_cell_tags_to_sub_mesh(sub_mesh, sf_parent):
    sf_submesh_out = MeshFunction('size_t', sub_mesh, 2)
    parent_cell_map = sub_mesh.data().array('parent_cell_indices', 2)

    # run over all cells of the sub_mesh
    for sub_cell in range(sub_mesh.num_entities(2)):
        # map the cell of the sub_mesh into the corresponding mesh of the parent cell
        parent_cell = parent_cell_map[sub_cell]
        # assign the correct id of the function sf_submesh_out calculated on the mesh of the sub_mesh under consideration, setting it to the same id it has in the parent_mesh
        sf_submesh_out[sub_cell] = sf_parent[parent_cell]

    return sf_submesh_out


'''
read a 1,2 or 3d mesh stored into an xdmf file
Input values: 
- 'input_path': the path where 'tetra_mesh.xdmf', 'triangle_mesh.xdmf', or 'line_mesh.xdmf' are located
Return values: 
- 'mesh': the mesh, or [] if the mesh could not be read
- 'sf': the mesh function for the components of the mesh with the largest dimension 
'''


def read_from_xdmf_file(mesh_path):
    mesh_path_with_slash = io.add_trailing_slash(mesh_path)

    if cmd.check_if_file_exists(mesh_path_with_slash + "tetra_mesh.xdmf"):
        mesh = read_mesh(mesh_path_with_slash + "tetra_mesh.xdmf")
        sf = read_mesh_components(mesh, mesh.topology().dim(), mesh_path_with_slash + "tetra_mesh.xdmf")
        print('3d mesh')

        result = mesh, sf

    else:
        if cmd.check_if_file_exists(mesh_path_with_slash + "triangle_mesh.xdmf"):
            mesh = read_mesh(mesh_path_with_slash + "triangle_mesh.xdmf")
            sf = read_mesh_components(mesh, mesh.topology().dim(), mesh_path_with_slash + "triangle_mesh.xdmf")

            print('2d mesh')

            result = mesh, sf

        else:
            if cmd.check_if_file_exists(mesh_path_with_slash + "line_mesh.xdmf"):
                mesh = read_mesh(mesh_path_with_slash + "line_mesh.xdmf")
                sf = read_mesh_components(mesh, mesh.topology().dim(), mesh_path_with_slash + "line_mesh.xdmf")

                print('1d mesh')

                result = mesh, sf
            else:

                print(f"{col.Fore.RED}No mesh could be loaded!{col.Style.RESET_ALL}")

                result = []

    return result


'''
read a 1d mesh stored into an h5 file
Input values: 
- 'mesh_path': the path where 'line_mesh.h5' is located
Return values: 
- 'mesh': the mesh, or [] if the mesh could not be read
- 'cf': the mesh function for the components of the mesh with the largest dimension (lines)
'''


def read_from_h5_file(mesh_path):
    mesh_path_with_slash = io.add_trailing_slash(mesh_path)

    if cmd.check_if_file_exists(mesh_path_with_slash + "line_mesh.h5"):
        mesh = read_mesh(mesh_path_with_slash + "line_mesh.h5")
        cf = read_mesh_components(mesh, mesh.topology().dim(), mesh_path_with_slash + "line_mesh.h5", "cf")

        result = mesh, cf

    else:
        print(f"{col.Fore.RED}No mesh could be loaded!{col.Style.RESET_ALL}")
        result = []

    return result


def read_from_file(mesh_path, file_format='xdmf'):
    if file_format == 'xdmf':
        return read_from_xdmf_file(mesh_path)
    elif file_format == 'h5':
        return read_from_h5_file(mesh_path)
    else:
        print(f"{col.Fore.RED}No mesh could be loaded!{col.Style.RESET_ALL}")


'''
write a mesh to xdmf file
Input values:
- 'mesh': the mesh
- 'map': the map containing the tags of the mesh elements (triangles, lines, etc) 
- 'output_file': path + name of the xdmf file where the mesh will be written 
'''


def write_mesh(mesh, output_file, map=None):
    with XDMFFile(output_file) as xdmf:
        xdmf.write(mesh)
        xdmf.write(map)
        xdmf.close()


'''
this method generates a submesh from a parent mesh
Input values:
- 'parent_mesh_path': the path where the field triangle_mesh.xdmf and line_mesh.xdmf are stored
- 'sub_mesh_path': the path where triangle_mesh.xdmf and line_mesh.xdmf of the submesh whill be stored
- 'sub_mesh_id' : the id with which the triangles of the submesh are tagged in the parent mesh

Return values:
- 'sub_mesh', 'boundary_sub_mesh': the sub_mesh and the mesh given by the boundary of the sub_mesh
'''


def generate_sub_mesh(parent_mesh_path, sub_mesh_path, sub_mesh_id):
    parent_mesh_path_slash = io.add_trailing_slash(parent_mesh_path)
    submesh_path_slash = io.add_trailing_slash(sub_mesh_path)

    parent_mesh = read_mesh(parent_mesh_path_slash + 'triangle_mesh.xdmf')

    # create entity maps fo the parent mesh
    sf_parent_mesh = read_mesh_components(parent_mesh, parent_mesh.topology().dim(), parent_mesh_path_slash + "triangle_mesh.xdmf")
    mf_parent_mesh = read_mesh_components(parent_mesh, parent_mesh.topology().dim() - 1, parent_mesh_path_slash + "line_mesh.xdmf")

    # extract the outer sub_mesh from the parent mesh, by picking only the triangles with submesh_id
    sub_mesh = SubMesh(parent_mesh, sf_parent_mesh, sub_mesh_id)
    # create the boundary mesh of sub_mesh
    sub_mesh_boundary = BoundaryMesh(sub_mesh, "exterior", order=True)

    # print(f'type of sub_mesh: {type(sub_mesh)}')

    # create entity maps of sub_mesh for triangles and lines
    sf_sub_mesh = transfer_cell_tags_to_sub_mesh(sub_mesh, sf_parent_mesh)
    mf_sub_mesh = transfer_facet_tags_to_sub_mesh(parent_mesh, sub_mesh, mf_parent_mesh)

    # create entity map for boundary mesh for lines
    mf_boundary_sub_mesh = transfer_facet_tags_to_bounday_mesh(sub_mesh_boundary, mf_sub_mesh)

    # write the triangles for sub_mesh to file
    write_mesh(sub_mesh, submesh_path_slash + "triangle_mesh.xdmf", sf_sub_mesh)
    # write the lines of the boundary mesh to file
    write_mesh(sub_mesh_boundary, submesh_path_slash + "line_mesh.xdmf", mf_boundary_sub_mesh)
    # print  submesh vertices to csv file
    io.print_mesh_vertices_to_csv(sub_mesh, submesh_path_slash + "vertices.csv")
    # print sub mesh metadata
    # io.write_parameters_to_csv_file(submesh_path_slash + "mesh_metadata.csv", submesh_parameters)

    return sub_mesh, sub_mesh_boundary


'''
generate a one-dimensional mesh as an IntervalMesh given its geometric parameters and tags
Input values: 
- 'x_l', 'x_r': the left and right x coordinate of the extremal points of the line mesh
- 'n_intervals': the number of intervals into which the line mesh is divided
- 'line_id': the id of the line mesh: all lien intervals will be tagged with this id
- 'vertex_l_id', 'vertex_r_id': the id of the extermal left and right vertices, respectively
- 'x_m_id' [optional]: the coordinate of the middle vertex in the mesh: this coordinate must match with one of the coordinates of the mesh vertices
- 'vertex_m_id': the id of the middle vertex in the mesh
- 'output_directory' [optional]: the path where the mesh will be written. In that path this method will write the mesh component, vertices and, if metadata != None, the mesh metadata
- 'metadata' [optional]: the mesh metadata to write in the output directory

Return values: 
- 'mesh': the one-dimensional mesh
- 'cell_function_temp': the mesh funciton tagging cells (line intervals) in the mesh
- 'vertex_function_temp': the mesh function tagging vertices in the mesh


Example of usage: 
          mesh_1d, cf_mesh_1d, vf_mesh_1d = msh.genereate_line_mesh(0, parameters['L'], len(x_coordinates) - 1,
                                                                                          parameters[f'sub_mesh_{p}_id'], parameters['vertex_sub_mesh_1_l_id'], parameters['vertex_sub_mesh_1_r_id'])
'''


def genereate_line_mesh(x_l, x_r, n_intervals, line_id, vertex_l_id, vertex_r_id, x_m=None, vertex_m_id=None, output_directory=None, metadata=None):
    
    mesh = IntervalMesh(n_intervals, x_l, x_r)

    # create a function for the lines
    cell_function = MeshFunction("size_t", mesh, mesh.topology().dim())
    cell_function.set_all(line_id)  # Tag entire line as region parameters['line_id']

    # creat a function for the vertices
    vertex_function = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    
    if (x_m is not None) and (vertex_m_id is not None):
        # I am generating a mesh with a middle vertex -> create the boolean variable vertex_m_exists to check whether x_m matches one of the coordinates of the mesh vertices
        vertex_m_exists = False
        
    for vertex in vertices(mesh):
        x = vertex.point().x()  # Get x-coordinate

        if math.isclose(x, x_l):
            vertex_function[vertex] = vertex_l_id

        if math.isclose(x, x_r):
            vertex_function[vertex] = vertex_r_id
            
        # if there is a middle vertex, tag id with vertex_m_id
        if (x_m is not None) and (vertex_m_id is not None):
            # I am generating a mesh with a middle vertex -> check if the mesh vertex coordinate under consideration matches x_m
             if math.isclose(x, x_m):
                vertex_function[vertex] = vertex_m_id
                vertex_m_exists = True

    if (x_m is not None) and (vertex_m_id is not None):
        # I am generating a mesh with a middle vertex -> if no vertex coordinate matches x_m, print an error message
        if vertex_m_exists is not True:
            print(f"{col.Fore.RED}{'Error: middle vertex is not one of the mesh vertices!'}{col.Style.RESET_ALL}")

    if output_directory is not None:
        '''
        write the mesh lines and vertices to .h5 files: 
        one needs to write them to .h5 file rather than to .xdmf file because only .h5 file can be properly read later on
        '''
        write_mesh_components_h5(mesh, output_directory + "line_mesh.h5", cell_function, "cf")
        write_mesh_components_h5(mesh, output_directory + "vertex_mesh.h5", vertex_function, "vf")

        io.print_mesh_vertices_to_csv(mesh, output_directory + "vertices.csv")

        # print mesh metadata
        if metadata is not None:
            io.write_parameters_to_csv_file(output_directory + "mesh_metadata.csv", metadata)

    return mesh, cell_function, vertex_function


'''
return the geometrical shape of an element for a mesh with different dimensions
Input values: 
- 'mesh': the mesh
Return values: 
- the geometry: 'tetrahedron' for a 3d mesh, 'triangle' for a 2d mesh, 'interval' for a 1d mesh

Example of usage:
    P_u = FiniteElement('P', msh.element_geometry(lmsh.mesh), rpam.parameters['function_space_degree'])
'''


def element_geometry(mesh):
    d = mesh.topology().dim()

    if d == 3:
        return tetrahedron
    elif d == 2:
        return triangle
    elif d == 1:
        return interval
