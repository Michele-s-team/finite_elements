from fenics import *
from mshr import *
from dolfin import *
import numpy as np


# this function prints a scalar field to csv file
def print_scalar_to_csvfile(f, filename):
    csvfile = open( filename, "w" )
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )
    for x, val in zip( f.function_space().tabulate_dof_coordinates(), f.vector().get_local() ):
        print( f"{val},{x[0]},{x[1]},{0}", file=csvfile )
    csvfile.close()


# this function prints a vector field to csv file
def print_vector_to_csvfile(f, filename):
    i = 0
    list_val_x = []
    list_val_y = []
    list_x = []
    for x, val in zip( f.function_space().tabulate_dof_coordinates(), f.vector().get_local() ):
        if (i % 2 == 0):
            list_val_x.append( val )
            list_x.append( x )
        else:
            list_val_y.append( val )

        i += 1

    csvfile = open( filename, "w" )
    print( f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile )

    for x, val_x, val_y in zip( list_x, list_val_x, list_val_y ):
        print( f"{val_x},{val_y},{0},{x[0]},{x[1]},{0}", file=csvfile )

    csvfile.close()


def print_in_bulk(f):
    Q = f.function_space()
    mesh = Q.mesh()

    print( f"Function space: {Q}" )

    vertex_function = MeshFunction( "size_t", mesh, 0 )

    class BoundaryMarker( SubDomain ):
        def inside(self, x, on_boundary):
            return on_boundary

    BoundaryMarker().mark( vertex_function, 1 )
    boundary_vertices = np.asarray( vertex_function.where_equal( 1 ) )

    print( f"dir = {dir( Q.dofmap() )}" )
    print( f"dofs = {len( Q.dofmap().tabulate_local_to_global_dofs() )}" )

    dofs = (Q.dofmap().tabulate_local_to_global_dofs())[boundary_vertices]

    print( f"dofs[boundary_vertices] 1= {(Q.dofmap().tabulate_local_to_global_dofs())[boundary_vertices]}" )
    #
    #
    # v_to_d = vertex_to_dof_map( Q )
    # print(f"dof[boundary_vertices]  2 = {v_to_d[boundary_vertices]}")

    # dofs = v_to_d[boundary_vertices]

    #
    x = Q.tabulate_dof_coordinates()
    for dof in dofs:
        print( x[dof], f.vector()[dof] )
