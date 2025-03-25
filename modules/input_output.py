import csv
from fenics import *
import mesh as msh

number_of_decimals = 2



#print the scalar field 'f' to csv file 'filename'
def print_scalar_to_csvfile(f, filename):
    csvfile = open( filename, "w" )
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )
    for x, val in zip( f.function_space().tabulate_dof_coordinates(), f.vector().get_local() ):
        print( f"{val},{x[0]},{x[1]},{0}", file=csvfile )
    csvfile.close()

#this function print a scalar defined only on the boundaries to csv file 
def print_scalar_boundary_to_csvfile(f, mesh, filename):
    csvfile = open( filename, "w" )
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )
    points = msh.boundary_points(mesh)
    for point in x:
        print( f"{f(point)},{point[0]},{point[1]},{0}", file=csvfile )
    csvfile.close()

#print the nodal values a scalar field 'f' on the mesh 'mesh' to csv file
def print_nodal_values_scalar_to_csvfile(f, mesh, filename):

    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q.tabulate_dof_coordinates()

    csvfile = open( filename, "w" )
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )

    for i in range( Q.dim() ):
        print( f"{f(coordinates[i][0], coordinates[i][1])}, {coordinates[i][0]}, {coordinates[i][1]}, {0}", file=csvfile )

    csvfile.close()


#print a vector field 'f' to csv file 'filename'
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


# this function prints a vector field defined on the boundary to csv file
def print_vector_boundary_to_csvfile(f, mesh, filename):
    points = msh.boundary_points(mesh)

    csvfile = open( filename, "w" )
    print( f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile )

    for point in points:
        print( f"{f(point[0],point[1])[0]},{f(point[0],point[1])[1]},{f(point[0],point[1])[2]},{point[0]},{point[1]},{0}", file=csvfile )

    csvfile.close()

#print the nodal values of a vector field 'f' on the mesh 'mesh' to csv file 'filename'
def print_nodal_values_vector_to_csvfile(f, mesh, filename):

    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q.tabulate_dof_coordinates()

    csvfile = open( filename, "w" )
    print( f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile )

    for i in range( Q.dim() ):
        v = f( coordinates[i][0], coordinates[i][1] )
        print( f"{v[0]}, {v[1]}, {0}, {coordinates[i][0]}, {coordinates[i][1]}, {0}", file=csvfile )

    csvfile.close()


#print to the csv file 'filename' the coordinates of the vertices of 'mesh'
def print_vertices_to_csv_file(mesh, filename):

    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q.tabulate_dof_coordinates()

    csvfile = open( filename, "w" )
    print( f"\":0\",\":1\"", file=csvfile )

    for i in range( Q.dim() ):
        print( f"{coordinates[i][0]}, {coordinates[i][1]}", file=csvfile )

    csvfile.close()


'''
read the tabulated  value of a scalar written in file 'filename' and return them as a table
table[i] = [value of the scalar at the ith vertex, x-coordinate of the i-th vertex, y coordinate of the ith vertex]
'''
def read_scalar_from_csvfile(filename):
    with open( filename, newline='', encoding='utf-8' ) as csvfile:
        reader = csv.reader( csvfile )
        next( reader )  # Skip the header row
        data = [[float( value ) for value in row] for row in reader]
    # print(data)
    return data

#if 'string' does not end by '/' add '/' to 'string'
def add_trailing_slash(string):
    if string[-1] != '/':
        return string + '/'
    else:
        return string