import colorama as col
import csv
from fenics import *
import os
import shutil

import mesh as msh

number_of_decimals = 2



#print the scalar field 'f' to csv file 'filename'
def print_scalar_to_csvfile(f, filename):

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csvfile = open( filename, "w" )
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )
    for x, val in zip( f.function_space().tabulate_dof_coordinates(), f.vector().get_local() ):

        padded_x = pad(x, 3)
        print( f"{val},{padded_x[0]},{padded_x[1]},{padded_x[2]}", file=csvfile )
    csvfile.close()



'''
print the nodal values a scalar field 'f' on the mesh 'mesh' to csv file
Input values: 
- 'f': the field
- 'mesh' the mesh 
- 'filename': the output filename
'''
def print_nodal_values_scalar_to_csvfile(f, mesh, filename):

    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q.tabulate_dof_coordinates()

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)


    csvfile = open( filename, "w" )
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )

    for i in range( Q.dim() ):

        coordinate = coordinates[i]
        # convert the coordinate in the correct format by addding 0s for the unused dimensions, in order to form an array of dimension 3
        padded_coordinate = pad(coordinate, 3)

        print( f"{f(*coordinate)}, {padded_coordinate[0]}, {padded_coordinate[1]}, {padded_coordinate[2]}", file=csvfile )

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

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csvfile = open( filename, "w" )
    print( f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile )

    for x, val_x, val_y in zip( list_x, list_val_x, list_val_y ):
        print( f"{val_x},{val_y},{0},{x[0]},{x[1]},{0}", file=csvfile )

    csvfile.close()



'''
print a vector field in the three-dimensional space to csv file 
Input values:
- 'V': the three-dimensional vector field, whcih returns a tuple of 3 values for each point in \Omega
- 'filename': the name of the csv file where 'V' will be written
'''
def print_vector_3d_to_csvfile(V, filename):

    i = 0
    list_val_x = []
    list_val_y = []
    list_val_z = []
    list_x = []

    for x, val in zip(V.function_space().tabulate_dof_coordinates(), V.vector().get_local()):
        if (i % 3 == 0):
            list_val_x.append(val)
            list_x.append(x)
        elif (i % 3 == 1):
            list_val_y.append(val)
        elif (i % 3 == 2):
            list_val_z.append(val)

        i += 1

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csvfile = open(filename, "w")
    print(f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile)

    for x, val_x, val_y, val_z in zip(list_x, list_val_x, list_val_y, list_val_z):
        print(f"{val_x},{val_y},{val_z},{x[0]},{x[1]},{0}", file=csvfile)

    csvfile.close()

#print the nodal values of a vector field 'f' on the mesh 'mesh' to csv file 'filename'
def print_nodal_values_vector_to_csvfile(f, mesh, filename):

    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q.tabulate_dof_coordinates()

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csvfile = open( filename, "w" )
    print( f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile )

    for i in range( Q.dim() ):

        coordinate = coordinates[i]
        # convert the coordinate in the correct format by addding 0s for the unused dimensions, in order to form an array of dimension 3
        padded_coordinate = pad(coordinate, 3)

        # convert the value of the vector field in the correct format by addding 0s for the unused dimensions, in order to form an array of dimension 3
        padded_f = pad(f( *coordinate ), 3)

        print( f"{padded_f[0]}, {padded_f[1]}, {padded_f[2]}, {padded_coordinate[0]}, {padded_coordinate[1]}, {padded_coordinate[2]}", file=csvfile )

    csvfile.close()

'''
print the nodal values of a vector field 'V', defined in the three-dimensional space in which the mesh and \Omega are embeedded,  
 to csv file 
 Input values:
- 'V': the three-dimensional vector field, whcih returns a tuple of 3 values for each point in \Omega
- 'mesh' : the mesh defining \Omega
- 'filename': the name of the csv file where 'V' will be written
'''
def print_nodal_values_vector_3d_to_csvfile(V, mesh, filename):

    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q.tabulate_dof_coordinates()

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csvfile = open( filename, "w" )
    print( f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile )

    for i in range( Q.dim() ):
        v = V(coordinates[i][0], coordinates[i][1])
        print( f"{v[0]}, {v[1]}, {v[2]}, {coordinates[i][0]}, {coordinates[i][1]}, {0}", file=csvfile )

    csvfile.close()


#print to the csv file 'filename' the coordinates of the vertices of 'mesh'
def print_vertices_to_csv_file(mesh, filename):

    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q.tabulate_dof_coordinates()

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csvfile = open( filename, "w" )
    print( f"\":0\",\":1\"", file=csvfile )

    for i in range( Q.dim() ):

        coordinate = coordinates[i]
        # convert the coordinate in the correct format by addding 0s for the unused dimensions, in order to form an array of dimension 3
        padded_coordinate = pad(coordinate, 3)


        print( f"{padded_coordinate[0]}, {padded_coordinate[1]}, {padded_coordinate[2]}", file=csvfile )

    csvfile.close()


'''
read the tabulated  value of a scalar defined on a 2d mesh, and  written in file 'filename' and return them as a table
table[i] = [value of the scalar at the ith vertex, x-coordinate of the i-th vertex, y coordinate of the ith vertex]
'''
def read_scalar_from_csvfile(filename):

    with open( filename, newline='', encoding='utf-8' ) as csvfile:
        reader = csv.reader( csvfile )
        next( reader )  # Skip the header row
        data = [[float( value ) for value in row] for row in reader]

    return data


#if 'string' does not end by '/' add '/' to 'string'
def add_trailing_slash(string):
    if string[-1] != '/':
        return string + '/'
    else:
        return string

'''
print a field as xdmf, h5, csv file and its nodal values on a csv file
Input values:
- 'f': the field
- 'path_xdmf_file' the path of the xdmf file
- 'path_csv_file' the path of the csv file
- 'path_h5_file' the path of the h5 file
- 'path_csv_nodal_value_file' the path of the csv file where the nodal values will be written
- 'mesh': the mesh where 'f' is defined
- 'type': the type of 'f', which may be 'scalar', 'vector' (for a vector in the tangent bundle of \Omega) or 'vector_3d' (for a vector in the 3d space in which 
\Omega is embedded)
'''
def full_print(f, field_name, path_xdmf_file, path_h5_file, path_csv_file, path_csv_nodal_value_file, mesh, type):

    # add / to file paths, in case it is missing
    path_xdmf_file_with_slash = add_trailing_slash(path_xdmf_file)
    path_h5_file_with_slash = add_trailing_slash(path_h5_file)
    path_csv_file_with_slash = add_trailing_slash(path_csv_file)
    path_csv_nodal_value_file_with_slash = add_trailing_slash(path_csv_nodal_value_file)

    # write to xdmf file
    xdmffile = XDMFFile(path_xdmf_file_with_slash + field_name + '.xdmf')
    xdmffile.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})
    xdmffile.write(f, 0)
    xdmffile.close()

    # write to h5 file
    HDF5File(MPI.comm_world, path_h5_file_with_slash + field_name + '.h5', "w").write(f, "/f")

    # write to csv file and the nodal values to csv file
    if type == 'scalar':
        print_scalar_to_csvfile(f, path_csv_file_with_slash + field_name + '.csv')
        print_nodal_values_scalar_to_csvfile(f, mesh, path_csv_nodal_value_file_with_slash + field_name + '.csv')

    elif type == 'vector':
        print_vector_to_csvfile(f, path_csv_file_with_slash + field_name + '.csv')
        print_nodal_values_vector_to_csvfile(f, mesh, path_csv_nodal_value_file_with_slash + field_name + '.csv')

    elif type == 'vector_3d':
        print_vector_3d_to_csvfile(f, path_csv_file_with_slash + field_name + '.csv')
        print_nodal_values_vector_3d_to_csvfile(f, mesh, path_csv_nodal_value_file_with_slash + field_name + '.csv')

'''
Print a text in red or green according to the value of a boolean variable. This function is used to print out tests
Input values:
- 'bool' : the boolean variable
- 'text': the text
'''
def check_print(bool, text_true, text_false):
    print(check_string(bool, text_true, text_false))

def check_string(bool, text_true, text_false):
    if bool:
        result = f'{col.Fore.GREEN}{text_true}{col.Fore.RESET}'
    else:
        result = f'{col.Fore.RED}{text_false}{col.Fore.RESET}'

    return result

# print a starred box of text 'message', in green if success = True and in red if success = False
def print_star_box(message, success=True):
    # Choose color
    color = col.Fore.GREEN if success else col.Fore.RED

    # Add spaces around the message
    message = f" {message} "

    # Width of the box
    box_width = len(message) + 8  # 4 spaces padding left and right inside box

    # Get terminal width
    terminal_width = shutil.get_terminal_size((80, 20)).columns  # fallback to 80 if unknown

    # Compute left padding to center the box
    left_padding = max((terminal_width - box_width) // 2, 0)  # no negative padding

    # Create top and bottom borders
    border = '#' * box_width

    # Build lines
    lines = [
        ' ' * left_padding + border,
        ' ' * left_padding + f"**{message.center(box_width - 4)}**",
        ' ' * left_padding + border
    ]

    # Print all lines with color
    for line in lines:
        print(color + line)

    print(col.Style.RESET_ALL, end='')  # Reset color after printing

'''
pad the array x with respect to a given dimension
Input values :
- 'x': the array, a list
- 'dim': the dimension
Return value:
- [x[0], x[1], ... , x[len(x)-1], 0, ...., 0] , an array of length 'dim'
'''
def pad(x, dim):
    return( list(x) + [0.0] * (dim - len(x)))
