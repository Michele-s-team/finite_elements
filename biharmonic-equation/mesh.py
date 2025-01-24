from fenics import *
import numpy as np


#  norm of vector x
def my_norm(x):
    return (sqrt( np.dot( x, x ) ))

class BoundaryMarker( SubDomain ):
    def inside(self, x, on_boundary):
        return on_boundary

# returns a list of boundary points of the mesh `mesh`
def boundary_points(mesh):
    # create a dummy function space of degree 1 which will be used only to extract the boundary points
    Q_dummy = FunctionSpace( mesh, 'CG', 1 )

    # a map which takes as an input a vertex of Q_dummy.mesh and returns its corresponding degree of freedom
    vertex_to_degree_of_freedom_map = vertex_to_dof_map( Q_dummy )

    # a function which takes as argument the mesh vertices
    vertex_function = MeshFunction( "size_t", mesh, 0 )

    vertex_function.set_all(0)

    # set vertex_function -> 1 on the vertices which are part of the boundary (vertex_function is zero elsewhere)
    BoundaryMarker().mark( vertex_function, 1 )
    # collect the vertices where the vertex_function = 1, i.e., the vertices on the boundary
    boundary_vertices = np.asarray( vertex_function.where_equal( 1 ) )

    degrees_of_freedom = vertex_to_degree_of_freedom_map[boundary_vertices]

    x = Q_dummy.tabulate_dof_coordinates()

    print("Degrees of freedom on the boundary:")
    for degree_of_freedom in degrees_of_freedom:
        print(f"\t{x[degree_of_freedom]}, {my_norm( x[degree_of_freedom])}")

    return x