import ast
import colorama as col
import csv
from fenics import *
import glob
import importlib
import numpy as np
import os
import pandas as pd
import shutil
import sys

import function as fu
msh = importlib.import_module('mesh.utils')

number_of_decimals = 2



'''
prints a field (scalar, vector, tensor) to csv file
Input values: 
    * Mandatory: 
        - 'f': the field
        - 'filename': path, filename and extension of the csv file
    * Optional:
        - 'mesh_function': a mesh function that tags regions of the mesh, needed if the function space of 'f' is discontinuous

Return values: 
    This method does not return anything but
    - If the function space of 'f' is a continuous one, the output csv file wil be of the form
        f:0,f:1,...,":0",":1",":2"
        f0_DOF0,f1_DOF0,...,rx_DOF_0,ry_DOF_0,rz_DOF_0
        f0_DOF1,f1_DOF1,...,rx_DOF_1,ry_DOF_1,rz_DOF_1
        ....
    - If the function space of 'f' is discontinuous
        f,":0",":1",":2",tag
        f0_DOF0,f1_DOF0,...,rx_DOF_0,ry_DOF_0,rz_DOF_0,tag_DOF_0
        f0_DOF1,f1_DOF1,...,rx_DOF_1,ry_DOF_1,rz_DOF_1,tag_DOF_1
        ...
'''

def print_to_csvfile(f, filename, mesh_function=None):

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    csvfile = open(filename, "w")

    Q = f.function_space()
    element  = Q.ufl_element()

    # value shape is the shape of 'f', for example () for a scalar, (3,) for a 3-component vector, and (2, 3) for a 2 x 3 tensor
    value_shape = element.value_shape()

    # value_size is the total number of components of 'f', for example for a (2, 3) tensor values_size = 2 * 3 
    value_size  = int(np.prod(value_shape)) if value_shape else 1

    # print(f'value_shape = {value_shape}\nvalue_size = {value_size}')

    mesh = Q.mesh()
    '''
    dof_coordinates stores the coordinates of the points where DOFs sit. Because the field 'f' defined on each DOF has value_size components, dof_coordinates is composed of blocks, where each block has 'value_size' entries, and blocks are all identical
    For example, dof_coordinates is of the form ->
        row 0:  [x0, y0]   ← this corresponds to 0th-component of f at DOF point 0
        row 1:  [x0, y0]   ← this corresponds to 1st-component of f at DOF point 0
        ...
        row value_size  [x1, y1]   ← this corresponds to 0th-component of f at DOF point 1
        row value_size+1 [x1, y1]   ← this corresponds to 1st-component of f at DOF point 1
        ...
        
    '''
    dof_coordinates = Q.tabulate_dof_coordinates()
    
    '''
    remove these identical entries by creating dof_coordinates_unique: I jump on 'dof_coordinates' with stride 'value_size' 
    dof_coordinates_unique = 
        [[x0, y0],
        [x1, y1],
        ....]
    '''
    dof_coordinates_unique = dof_coordinates[::value_size]

    '''
    f_values contains the value of 'f' on the DOFs, and it has the same structure as 'dof_coordinates'
    f_values = 
            entry 0: f[0] at DOF point 0
            entry 1: f[1] at DOF point 0
            ...
            entry value_size  f[0] at DOF point 1
            entry value_size + 1 f[1] at DOF point 1
            ...
        
    '''
    f_values = f.vector().get_local()

    '''
    f_values_unique is obtained by f_values by putting f[0], f[1], ... into a nested list
    f_values_unique = 
            [f[0] at DOF point 0, f[1] at DOF point 0, ...]
            [f[0] at DOF point 1, f[1] at DOF point 1, ...]
            ...
    
    '''
    f_values_unique = f_values.reshape(-1, value_size)

    # print(f'len dof_coordinates = {len(dof_coordinates)}')
    # print(f'len dof_coordinates_unique = {len(dof_coordinates_unique)}')
    # print(f'len f_values = {len(f_values)}')

    if (f.function_space().ufl_element().family() == 'Discontinuous Lagrange'):
        # the methods has been called with a discontinuous function space

        Q_continuous = False
        
        if mesh_function == None:
            # the meshod has been called on a discontinuous function space and mesh_function has not been provided -> 'mesh_function' is needed to tell to which tagged domain each DOF coordinate corresponds -> throw an error and exit

            print(f'{col.Fore.RED}Error!! print_to_csv_file has been called on a discontinuous function space without providing mesh_function.{col.Fore.RESET}')
            sys.exit(1)

    else: 
        # the method has been called with a continuous function space

        Q_continuous = True

    if value_size == 1:
        component_headers = '\"f\"'
    else:
        component_headers = ",".join([f'"f:{i}"' for i in range(value_size)])
        
    coordinate_headers = ",".join([f'":{i}"' for i in range(3)])

    headers = f'{component_headers},{coordinate_headers}'

    if Q_continuous == False:

        headers += ',tag' 

   
    print(headers, file=csvfile)

    if Q_continuous:
        # the function space of 'f' is continuous -> run over all unique DOF coordinates and print the value of f[0], f[1], ...  on each of them
        
        for dof_coordinate, f_value in zip(dof_coordinates_unique, f_values_unique):

            # pad 'x' to three dimensions
            x = pad(dof_coordinate, 3)

            f_value_string = ",".join([str(list(f_value)[i]) for i in range(value_size)])

            print(f"{f_value_string},{x[0]},{x[1]},{x[2]}", file=csvfile)
        
    
    else:
        # the function space of 'f' is discontinuous: the same DOF coordinate and f value may appear multiple times in f_values, each time for a different cell (and possibly mesh region) to which it belongs -> print the values of 'f' by looping through cells
        
        '''
        counter = 0
        target = np.array([0.3, 0.5])
        for cell in cells(mesh):
            cell_dofs = Q.dofmap().cell_dofs(cell.index())
            coords = dof_coordinates[cell_dofs]
            for row in coords:
                diff = np.linalg.norm(row - target)
                if diff < 1e-10:
                    print(f'cell {cell.index()}: found target, diff = {diff}, exact coords = {repr(row)}')
                    counter+=1

        print(f'found that coordinate {counter} times')
        '''

        for cell in cells(mesh):
            # run over all cells in 'mesh'

            #compute 'mesh_function' on the cell under consideration to tell to which tagged domain the cell under consideration belongs
            cell_tag = mesh_function[cell]

            '''
            cell_dofs contains the IDs of the DOFs that are contained into 'cell', it has the structure
            [
                id_f_0_on_DOF_0, 
                id_f_0_on_ DOF_1,
                ...,
                id_f_0_on_DOF_{n_nodes-1},

                id_f_1_on_DOF_0, 
                id_f_1_on_ DOF_1,
                ...,
                id_f_1_on_DOF_{n_nodes-1},

                ...
            ]
            where the pattern is repeated value_size times, i.e., one for each component of 'f', and n_nodes = [number of DOFs in the cell] / [value_size]. In other words

            cell_dofs[j * n_nodes + i] = [index in f.values().get_local() corresponding to the j-th component of the tensor 'f' sitting on ith DOF in the cell 'cell']
            '''
            cell_dofs = Q.dofmap().cell_dofs(cell.index())


            dof_coordinates_cell = dof_coordinates[cell_dofs]

            n_nodes = len(cell_dofs) // value_size

            # print(f'cell_dofs = {cell_dofs}')

            '''
            remove the redundancy in cell_dofs and store the result in 
            cell_dofs_unique = 
                [
                    id_DOF_0, 
                    id_DOF_1,
                    ...,
                    id_DOF_{n_nodes-1}
                ]
            
            '''
            cell_dofs_unique = cell_dofs[:n_nodes]

            # print(f'dof coorindates in cell: {dof_coordinates_cell}')

            for i in range(len(cell_dofs_unique)):
                # run over physical DOFs contained to 'cell' and print out the value of 'f' by specifying that those DOFs belong to region tagged with 'cell_tag' in a separate column of the csv output file. Note that, because the space of 'f' is discontinuous, here DOFs in 'cell' may belong to different mesh regions, and thus have different tags

                # pad 'x' to three dimensions
                dof_coordinate = pad(dof_coordinates[cell_dofs_unique[i]], 3)

                f_value_string = ",".join([f'{f_values[cell_dofs[j * n_nodes + i]]}' for j in range(value_size)])
                dof_coordinate_string = ",".join([f'{dof_coordinate[j]}' for j in range(len(dof_coordinate))])

                print(f"{f_value_string},{dof_coordinate_string},{cell_tag}", file=csvfile)


    csvfile.close()



'''
print the nodal values of a field (scalar, vector or tensor) to csv file
Input values: 
    - 'f': the field
    - 'filename': the path, filename and extension of the csv file where the tensor will be written 

Return values: 
This method does not return anythinb, but the resulting csv file is of this form

    f:0,f:1,....,f:[number of components of f],:0,:1,:2
    t_0,t_1,....,t_[(number of components of f) - 1],x_0,x_1,x_2
    ....

The header is 'f,:0,:1,:2' is 'f' is a scalar. 
'''

def print_nodal_values_to_csvfile(f, filename):

    if (f.function_space().ufl_element().family() != 'Discontinuous Lagrange'):
        # 'f' is not defined on a DG function space -> proceed

        # create the path for the csv file if it does not exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        Q = f.function_space()
        mesh = Q.mesh()
        
        # the shape of the tensor, for example (2, 3)
        value_shape = Q.ufl_element().value_shape()
        
        # value_size is the total number of components of 'f', for example is 'f' is a (2, 3) tensor, shape_size = 2 * 3 
        value_size  = int(np.prod(value_shape)) if value_shape else 1

        # a dummy function space of order 1 used to tabulated the vertices
        Q_1 = FunctionSpace(mesh, 'CG', 1)
        coordinates = Q_1.tabulate_dof_coordinates()

        csvfile = open(filename, "w")

        if value_size == 1:
            component_headers = '\"f\"'
        else:
            component_headers = ",".join([f'"f:{i}"' for i in range(value_size)])

        coord_headers = ",".join([f'":{i}"' for i in range(3)])

        print(f"{component_headers},{coord_headers}", file=csvfile)

        for i in range(Q_1.dim()):
            # run through the mesh nodes

            coordinate = coordinates[i]

            # convert the coordinate in the correct format by addding 0s for the unused dimensions, in order to form an array of dimension 3
            padded_coordinate = pad(coordinate, 3)

            # evaluate the field at the coordinate
            # if the field has only one component, atleast_1d converts it to an array with one entry so it has the correct format 
            f_value = np.atleast_1d(f(*coordinate))

            component_str = ",".join([str(f_value[j]) for j in range(value_size)])
            coord_str     = f"{padded_coordinate[0]},{padded_coordinate[1]},{padded_coordinate[2]}"

            print(f"{component_str},{coord_str}", file=csvfile)


        csvfile.close()

    else:
        # 'f' is defined on a discontinuous function space -> it does not make sense to call this method because f(x) may be ill-defined -> throw an error and exit

        print(f'{col.Fore.RED}Error!! print_nodal_values_to_csvfile has been called on a discontinuous function space. Stopping now. {col.Fore.RESET}')
        sys.exit(1)
    

'''
print the coordinates of the vertices of a mesh to csv file
Input values: 
- 'mesh' <dolfin.Mesh>: the mesh
- 'outfile': path of the csv file
'''


def print_mesh_vertices_to_csv(mesh, filename):
    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace(mesh, 'CG', 1)
    coordinates = Q.tabulate_dof_coordinates()

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csvfile = open(filename, "w")
    print(f"\":0\",\":1\",\":2\"", file=csvfile)

    for i in range(Q.dim()):
        coordinate = coordinates[i]
        # convert the coordinate in the correct format by addding 0s for the unused dimensions, in order to form an array of dimension 3
        padded_coordinate = pad(coordinate, 3)

        print(f"{padded_coordinate[0]}, {padded_coordinate[1]}, {padded_coordinate[2]}", file=csvfile)

    csvfile.close()


def print_mesh_triangles_to_csv(mesh, filename):
    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    csvfile = open(filename, "w")
    # Header with coordinate labels for 3 vertices
    print(f"\"p_1:0\",\"p_1:1\",\"p_1:2\",\"p_2:0\",\"p_2:1\",\"p_2:2\",\"p_3:0\",\"p_3:1\",\"p_3:2\"", file=csvfile)

    # Iterate through all cells (triangles) in the mesh
    for cell in cells(mesh):
        # Get the coordinates of the three vertices of this triangle
        vertices = cell.get_vertex_coordinates()
        
        # pad each vertex with z=0
        p1 = pad(vertices[0:2], 3)
        p2 = pad(vertices[2:4], 3)
        p3 = pad(vertices[4:6], 3)
        
        print(f"{p1[0]},{p1[1]},{p1[2]},{p2[0]},{p2[1]},{p2[2]},{p3[0]},{p3[1]},{p3[2]}", file=csvfile)

    csvfile.close()


'''
print the coordinates of the extermal points of the lines of a mesh to csv file
Input values: 
- 'mesh' <dolfin.Mesh>: the mesh
- 'outfile': path of the csv file
'''


def print_mesh_lines_to_csv(mesh, outfile):
    """
    Export unique edges of a FEniCS mesh to CSV with 3D coordinates (padded using np.pad).
    Compatible with 1D, 2D, and 3D meshes.
    """

    mesh.init()  # Ensure all connectivities exist

    # Ensure edge-to-vertex connectivity exists
    try:
        mesh.init(1, 0)
    except RuntimeError:
        pass  # Already initialized

    coordinates = mesh.coordinates()
    gdim = mesh.geometry().dim()

    edge_set = set()
    for edge in edges(mesh):
        v = edge.entities(0)
        edge_set.add(tuple(sorted(v)))

    with open(outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["start:0", "start:1", "start:2", "end:0", "end:1", "end:2"])

        for v_start, v_end in sorted(edge_set):
            p_start = coordinates[v_start]
            p_end = coordinates[v_end]

            # Pad to 3D
            p_start_padded = np.pad(p_start, (0, 3 - len(p_start)), mode='constant')
            p_end_padded = np.pad(p_end, (0, 3 - len(p_end)), mode='constant')

            writer.writerow(np.concatenate([p_start_padded, p_end_padded]))


'''
read the tabulated  value of a scalar defined on a 2d mesh, and  written in file 'filename' and return them as a table
table[i] = [value of the scalar at the ith vertex, x-coordinate of the i-th vertex, y coordinate of the ith vertex, z coordinate of the ith vertex]
'''


def read_scalar_from_csvfile(filename):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        data = [[float(value) for value in row] for row in reader]

    return data


# if 'string' does not end by '/' add '/' to 'string'
def add_trailing_slash(string):
    if string[-1] != '/':
        return string + '/'
    else:
        return string

'''
print a field to xdmf file
Input values: 
- 'f': the field (scalar, vector, or tensor)
- 'path': the full path, including file name and extension, of the file
'''

def xdmf_print(f, path):
    # write to xdmf file
    xdmffile = XDMFFile(path)
    xdmffile.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})
    xdmffile.write(f, 0)
    xdmffile.close()


'''
return the type of a field (scalar, vector or tensor) as a string
Input values: 
    - 'f': the field
Return values: 
    - 'type': 'scalar', 'vector' or 'tensor', if f is a scalar, vector or tensor field, respectively
'''
def field_type(f):

    shape = f.function_space().ufl_element().value_shape()

    if len(shape) == 0:
        return 'scalar'
    elif len(shape) == 1:
        return 'vector'
    else:
        return 'tensor'


'''
print a field as xdmf, h5, csv file and its nodal values on a csv file
Input values:
    * Mandatory:
        - 'f': the field
        - 'path_xdmf_file' the path of the xdmf file
        - 'path_csv_file' the path of the csv file
        - 'path_h5_file' the path of the h5 file
        - 'path_csv_nodal_value_file' the path of the csv file where the nodal values will be written
    * Optional:
        - 'mesh_function': a mesh function that tags mesh region, needed to plot fields on discontinuous spaces
Return values:
    This method does not return anything, but it writes 'f' to xdmf, h5 files. It also writes the DOF values of 'f' to csv file, and, if 'f' is defined on a continuous function space, the nodal values of 'f' on mesh vertices. 
'''


def full_print(f, field_name, path_xdmf_file, path_h5_file, path_csv_file, path_csv_nodal_value_file,
               mesh_function=None):

    type = field_type(f)

    # add / to file paths, in case it is missing
    path_xdmf_file_with_slash = add_trailing_slash(path_xdmf_file)
    path_h5_file_with_slash = add_trailing_slash(path_h5_file)
    path_csv_file_with_slash = add_trailing_slash(path_csv_file)
    path_csv_nodal_value_file_with_slash = add_trailing_slash(path_csv_nodal_value_file)

    # write to xdmf file
    xdmf_print(f, path_xdmf_file_with_slash + field_name + '.xdmf')

    # write to h5 file
    hdf5_file = HDF5File(MPI.comm_world, path_h5_file_with_slash + field_name + '.h5', "w")
    hdf5_file.write(f, "/f")
    hdf5_file.close()

    # write to csv file 
    print_to_csvfile(f, path_csv_file_with_slash + field_name + '.csv',
                                mesh_function=mesh_function)
    
    if (f.function_space().ufl_element().family() != 'Discontinuous Lagrange'):
        # the field is defined on a continuous space -> print its nodal values to csv file

        print_nodal_values_to_csvfile(f, path_csv_nodal_value_file_with_slash + field_name + '.csv')


'''
print on a target mesh a field (scalar, vector or tensor) defined on a source mesh, where the source and target mesh are related by a deformation field
Input values: 
    * Mandatory:
        - 'f': the field
        - 'u': the deformation field, defined on the source mesh
        - 'path_xdmf_file' the path of the xdmf file
        - 'path_h5_file' the path of the h5 file
        - 'path_csv_file' the path of the csv file
        - 'path_csv_nodal_value_file' the path of the csv file where the nodal values of the field will be written 
     * Optional:
        - 'mesh_function': a mesh function that tags mesh region, needed to plot fields on discontinuous spaces
'''
def full_print_deformed(f, u, field_name, path_xdmf_file, path_h5_file, path_csv_file, path_csv_nodal_value_file, mesh_function=None):

    f_def = fu.deform_function(f, u)

    full_print(f_def, 'def_' + field_name, path_xdmf_file, path_h5_file, path_csv_file, path_csv_nodal_value_file, mesh_function=mesh_function)


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


# Also need to update the pad function to be more robust:
def pad(x, dim):
    # Handle the case where x might be a scalar
    if hasattr(x, '__iter__'):
        return (list(x) + [0] * (dim - len(x)))
    else:
        # x is a scalar, treat as single-element list
        return ([x] + [0] * (dim - 1))

'''
count the number of files which match a given path pattern
Input values :
- 'path_before_asterisk', 'path_after_asterisk': the path before and after asterisk
Ouput values: 
- the number of files matching that path

Example of usage:
To count all files  /home/fenics/shared/dynamics/channel_with_cylinder_flat_icps/solution/snapshots/csv/nodal_values/u_n_*.csv do
    count_files('/home/fenics/shared/dynamics/channel_with_cylinder_flat_icps/solution/snapshots/csv/nodal_values/u_n_', '.csv')
'''


def count_files(path_before_asterisk, path_after_asterisk):
    return len(glob.glob(path_before_asterisk + '*' + path_after_asterisk))

'''
Convert a string containing a numerical value to a number
Input values :
- 'string': the string containing the value (it may be an int, a float or a list)

Example of usage:
    string_to_value('13')
    string_to_value('2.43')
    string_to_value('[1,2]')
'''
def string_to_value(value):
    value = value.strip()

    # check whether 'value' is a list
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Fallback: return as string
    return value

'''
read a set of parameters in a csv file
Input values:
- 'file_path': the path of the file
- 'parameter_name': the name of the parameter to be read (the name of one of the columns in the csv file)
Return value:
- the value of the parameter
'''


def read_parameter_from_csv_file(file_path, parameter_name, return_type=float):
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        row = next(reader)  # jump the first row with parameter names
        return return_type(row[parameter_name])


'''
write a list of parameters to csv file
Input values:
- 'file_path': the path of the file, including file name and extension
- 'parameters': the list of parameter names and values

Example of usage:
    write_parameters_to_csv_file('/home/my_nice_file.csv', [('L', 0.4334), ('x_p', 2.23), ('resolution', 0.01)])
'''
def write_parameters_to_csv_file(file_path, parameters):

    print(f'Writing parameters to {file_path}...', flush=True)

    # create the folder if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # remove the output file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    output_file = open(file_path, 'w', newline='')

    parameter_names = list(parameters.keys())

    # write to file
    writer = csv.DictWriter(output_file, fieldnames=parameter_names)

    writer.writeheader()
    writer.writerow(parameters)

    output_file.close()

    print('... done.', flush=True)


'''
Read a set of parameters from a csv file
Input values:
    * Mandatory:
        - 'file_path': the path of the file, including file name and extension
    * Optional:
        - 'print_out': if 'True' the read parameters are printed out. They will not be printed out otherwise. 

Return values:
    - the list of parameter names and values, e.g., [('L', 0.4334), ('x_p', 2.23), ('resolution', 0.01)]
'''
def read_parameters_from_csv_file(file_path, print_out=False):

    print(f'Reading parameters from {file_path}...',flush=True)

    file = open(file_path, newline='')

    reader = csv.reader(file)

    parameter_names = next(reader)
    parameter_values = next(reader)

    # print(f'parameter_names: {parameter_names}')
    # print(f'parameter_values: {[string_to_value(parameter_value) for parameter_value in parameter_values]}')

    file.close()
    print('... close.',flush=True)

    result = dict([(parameter_name, string_to_value(parameter_value)) for parameter_name, parameter_value in zip(parameter_names, parameter_values)])
    if print_out:
        print(f'Read parameters : {result}.',flush=True)

    return result


'''
merge two dictionaries 
Input values: 
- 'dictionary_a', 'dictionary_b': the two dictionaries to be merged
Return values: 
- the merged dictionary
'''
def merge_dictionaries(dictionary_a, dictionary_b):
    return {**dictionary_a, **dictionary_b}

'''
return the maximal among the values in a dictionary
Input values: 
- 'dictionary_a': the dictionary
Return values: 
- the maximal value
'''
def max_dictionary(dictionary):

    max = list(dictionary.values())[0]
    for key, value in dictionary.items():
        if value > max:
            max = value

    return max


'''
read a function expression provided as a string and return a function, that can be evaluated, corresponding to it
Input values:
    - 'function_string': the string corresponding to the function, such as 'function_string' = [cos(t), 2*sin(t), ...]    
Return values: 
    - the corresponding function [2*np.cos(t), 2*np.sin(t), ... ]
'''
def read_function_expresssion(function_string):

    # remove whitespaces
    function_string = function_string.strip()

    # expect format [expr_0, expr_1, ...]
    if not (function_string.startswith('[') and function_string.endswith(']')):
        raise ValueError(f"shape_parametric_form must be of the form [expr_0, expr_1, ...], got: {function_string!r}")
    
    function_string = function_string[1:-1]  # remove [ and ]


    parts = [p.strip() for p in function_string.split(',')]
    if any(p == '' for p in parts):
        raise ValueError(f"shape_parametric_form could not be parsed: {function_string!r}")

    # numpy_names are all the names defined in numpy, such as 'cos', 'log', ... 
    numpy_names = dir(np)

    # run through all numpy names, e.g., 'cos' and get the function associated with that numpy name, and store it into numpy_functions
    numpy_functions = {k: getattr(np, k) for k in numpy_names if not k.startswith('_')}
    numpy_functions['__builtins__'] = {}


    # define a function from the string expr_string that associates with 'cos' in the string the function np.cos, etc...
    def f(t):
        env = {**numpy_functions, 't': np.asarray(t, dtype=float)}
        return [eval(expr, env) for expr in parts]

    return f


'''
read a DG field (scalar, vector or tensor) from file. This works only if the field has been written to csv file with the method `print_to_csvfile` 
Input values; 
    - 'filepath': path, filename and extension of the csv file
    - 'f': the field in which the result will be written
'''
def read_dg_field_from_csv_file(filepath, f):

    print(f'path = {filepath}')

    # function space, value_shape, value_size and mesh relative to the field 'f'
    Q          = f.function_space()
    value_shape = Q.ufl_element().value_shape()
    value_size  = int(np.prod(value_shape)) if value_shape else 1
    mesh        = Q.mesh()

    # read the csv file and store it in a pandas data frame
    f_data = pd.read_csv(filepath)

    # match column names used in print_to_csvfile
    column_names = ['f'] if value_size == 1 else [f'f:{j}' for j in range(value_size)]

    # vector of DOF values of the field 'f'
    f_values  = f.vector().get_local()


    row_idx = 0
    for cell in cells(mesh):
        # loop through cells in the mesh

        '''
        cell_dofs contains the IDs of the DOFs that are contained into 'cell', it has the structure
        [
            id_f_0_on_DOF_0, 
            id_f_0_on_ DOF_1,
            ...,
            id_f_0_on_DOF_{n_nodes-1},

            id_f_1_on_DOF_0, 
            id_f_1_on_ DOF_1,
            ...,
            id_f_1_on_DOF_{n_nodes-1},

            ...
        ]
        where the pattern is repeated value_size times, i.e., one for each component of 'f', and n_nodes = [number of DOFs in the cell] / [value_size]. In other words, 
        cell_dofs[j * n_nodes + i] = [index in f.values().get_local() corresponding to the j-th component of the tensor 'f' sitting on ith DOF in the cell 'cell']
        '''
        cell_dofs = Q.dofmap().cell_dofs(cell.index())

        n_nodes   = len(cell_dofs) // value_size

        for i in range(n_nodes):
            # loop through nodes in `cell`

            for j in range(value_size):
                # loop through components of `f`

                # write into f_values the value of the field by extracting the row and column in f_data, i.e., in the csv file, following the same structure in which the field has been written to csv file by `print_to_csvfile`
                f_values[cell_dofs[j * n_nodes + i]] = f_data.iloc[row_idx][column_names[j]]

            row_idx += 1

    # write f_values into f.vector
    f.vector()[:] = f_values
    f.vector().apply('insert')

