from fenics import *
import numpy as np
import colorama as col
import gmsh
import meshio

import calc as cal
import geometry as geo
import input_output as io


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh


# read the mesh form 'filename' and return it
def read_mesh(filename):
    mesh = Mesh()

    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mesh)
    xdmf.close()

    return mesh


'''
read the mesh  from  the .msh file 'infile' and write the mesh components (tetrahedra, triangles, lines, vertices) to 'outfile' (tetrahedron_mesh.xdmf, triangle_mesh.xdmf ...)
the component type can be "tera", "triangle", "line" or "vertex"
if 'prune_z' = true (false), the z component will be removed from the mesh
'''


def write_mesh_components(infile, outfile, component_type, prune_z):
    mesh_from_file = meshio.read(infile)
    component_mesh = create_mesh(mesh_from_file, component_type, prune_z)
    meshio.write(outfile, component_mesh)


'''
given a mesh 'mesh', read its components of dimension 'dim' stored into 'filename' and returns the collection of components
Example: to read the lines of the mesh, call this method with 
cf = msh.read_mesh_components(mesh, 1, (args.input_directory) + "/line_mesh.xdmf")
'''


def read_mesh_components(mesh, dim, filename):
    mesh_value_collection = MeshValueCollection("size_t", mesh, dim)
    with XDMFFile(filename) as infile:
        infile.read(mesh_value_collection, "name_to_read")
        infile.close()
    return cpp.mesh.MeshFunctionSizet(mesh, mesh_value_collection)


# compare the numerical value of the integral of a test function over a ds, dx, .... with the exact one and output the relative difference
def test_mesh_integral(exact_value, f_test, measure, label):
    numerical_value = assemble(f_test * measure)
    print(
        f"{label} = {numerical_value:.{4}}, should be {exact_value:.{4}}, relative error =  {col.Fore.YELLOW}{abs((numerical_value - exact_value) / exact_value):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


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
    # print(f"\t{x[degree_of_freedom]}, {geo.my_norm( x[degree_of_freedom])}")

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
    # print(f"\t{x[degree_of_freedom]}, {geo.my_norm( x[degree_of_freedom])}")

    return x


# return the set of boundary points whose distance from the point c lies between r and R
def boundary_points_circle(mesh, r, R, c):
    points = boundary_points(mesh)

    x = []
    for point in points:
        if ((geo.my_norm(point - c) > r) and (geo.my_norm(point - c) < R)):
            x.append(point)

    # csvfile = open( "test_boundary_points_circle.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    return x


# compute the lowest and largest x and y values of points in the mesh and return them as a vector in the format [[x_min, x_max], [y_min, y_max]]
def extremal_coordinates(mesh):
    points = boundary_points(mesh)
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
write to csv file 'outfile' the coordinates of the start and end vertices which define the lines of the triangles of the mesh in the .msh file 'infile'
the vertices are written in the format
edge1_start[0], edge1_start[1], edge1_start[2], edge1_end[0], edge1_end[1], edge1_end[2]
edge2_start[0], edge2_start[1], edge2_start[2], edge2_end[0], edge2_end[1], edge2_end[2]
...
'''


def write_mesh_to_csv(infile, outfile):
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


# tag the objects of dimension 'dimension' contained in 'list_of_objects = [object_1, object_2, ...]' as physical objects, all with tag 'id' and with label 'label'
def tag_group(list_of_objects, dimension, id, label):
    gmsh.model.addPhysicalGroup(dimension, list_of_objects, id)
    gmsh.model.setPhysicalName(dimension, id, label)
