'''
This Code generates a symmetric square mesh.
If you want to generate the mesh form the terminal use :
    python generate_mesh.py <resolution> <output_dir>
    where resolution is the mesh size and output_dir is the directory where to save the mesh
    The half mesh will be saved in the output_dir as mesh.msh, while the complete mesh as mesh.xdmf
    The mesh will be saved in the output_dir as line_mesh.xdmf and triangle_mesh.xdmf
'''

import meshio #for reading and writing mesh files
import gmsh #main tool
import pygmsh #wrapper for gmsh
import argparse
import sys
import numpy as np

# add the path where to find the shared modules
module_path = '/home/tanos/Thesis/finite_elements/modules/'
sys.path.append( module_path )
import mesh as msh


parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_dir")
args = parser.parse_args()

#mesh resolution
resolution = (float)(args.resolution)
r = 1
L = 30
h = 30
c_r = [0, 0, 0]
c_R = [0, 0, 0]



output_dir = args.output_dir
mesh_file = output_dir + "/mesh.msh"

'''This function duplicates and transform all points, inverting their position with respect to the x axis
-points : Array of points to be duplicated
-point_data : Data that contains dimensional tag of the points (must be duplicated as well to avoid issues during the reading of the mesh)
Outputs
-new_points : the old and the new points
-new-points_indices : the indices of the new points in the new array (they are not just the indices of the old points traslated by some constant since the points on the x axis has not been duplicated and they were not ordered in the old list)
-mirrored_point_data : new point_data array
'''

def mirror_points(points, point_data):
        offset = 0
        new_points_indices = []
        mirrored_points = []
        mirrored_point_data = []
        for i in range(len(points)):
            if np.isclose(points[i,1] , 0, rtol=1e-3):
                offset += 1
                new_points_indices.append(i)       
            else:
                new_points_indices.append(i-offset+len(points))
                l = list(point_data['gmsh:dim_tags'][i,:])
                mirrored_point_data.append(l)
                # Flip the y-coordinates
                mirrored_points.append([points[i,0], points[i, 1]*-1, points[i,2]])

        mirrored_points = np.array(mirrored_points)
        new_points = np.vstack((points, mirrored_points))
        return new_points, new_points_indices, mirrored_point_data

'''
Half mesh is generated used pygmsh and it's saved as mesh.msh
'''
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()


N = int(np.round(np.pi/(resolution)))
half_rectangle_points = [model.add_point( (L/2, 0, 0), mesh_size=resolution*(min(L,h)/r) ),
            model.add_point( (L/2, h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
            model.add_point( (-L/2, h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
            model.add_point( (-L/2, 0, 0), mesh_size=resolution*(min(L,h)/r) ),
            ]

half_circle_points = [model.add_point( (-r*np.cos(np.pi*i/N), r*np.sin(np.pi*i/N), 0), mesh_size=resolution ) for i in range(N+1)]
my_points = half_rectangle_points + half_circle_points
channel_lines = [model.add_line( my_points[i], my_points[i + 1] )
                for i in range( -1, len( my_points ) - 1 )]

channel_loop = model.add_curve_loop( channel_lines )
plane_surface = model.add_plane_surface( channel_loop )

model.synchronize()

model.add_physical( [plane_surface], "Volume" )
model.add_physical( [channel_lines[1]], "r" )
model.add_physical( [channel_lines[3]], "l" )
model.add_physical( [channel_lines[2]], "t" )
#model.add_physical( [channel_lines[4],channel_lines[0]], "b" )
model.add_physical( channel_lines[5:], "c" )

geometry.generate_mesh( dim=2 )
gmsh.write( mesh_file)

#msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

gmsh.clear()
geometry.__exit__()


'''This part duplicate points and cells with the respective tags and ids
The new mesh inherits the ids (physical id used for measure definito) of the original one, except for the new physical objects that are generated from reflection (e.g. the b line)
'''
# in particular the rule 4:5 implies that the lines that in the original mesh where in the physical group 4 (top lines), when reflected, they will be assigned the id 5 (used to define measure in the bottom line)
ids = [0, 1, 2, 3, 5, 6] #{1:1, 2:2, 3:3, 4:5, 5:6} 
# Load the half-mesh
mesh = meshio.read(mesh_file)
print("original points", np.shape(mesh.points))

# Mirror points across X=0
new_points, new_points_indices, new_point_data = mirror_points(mesh.points, mesh.point_data)

original_triangles = mesh.cells_dict['triangle']
original_lines = mesh.cells_dict['line']

#duplicate cell blocks of type 'triangle'
triangles = np.copy(original_triangles)
for i in range(np.shape(triangles)[0]):
    for j in range(3):
        triangles[i,j] = new_points_indices[triangles[i,j]]
mesh.points = new_points
mesh.point_data['gmsh:dim_tags'] = np.vstack((mesh.point_data['gmsh:dim_tags'], new_point_data))
mesh.cells[-1] = meshio.CellBlock("triangle", np.vstack((original_triangles, triangles)))
print(mesh.cells[-1])
N = np.shape(mesh.cells[-1].data)[0]
mesh.cell_data['gmsh:physical'][-1] = np.array([mesh.cell_data['gmsh:physical'][-1][0]]*N)
mesh.cell_data['gmsh:geometrical'][-1] = np.array([mesh.cell_data['gmsh:geometrical'][-1][0]]*N)


#duplicate cell blocks of type 'line'
for j in range(len(mesh.cells)):
    if mesh.cells[j].type == 'line':
        lines = np.copy(mesh.cells[j].data)
        filtered_lines = []
        for i in range(np.shape(lines)[0]):
            f = [mesh.points[lines[i,k]][1]!= 0 for k in range(2)] 
            if f[0] or f[1]:
                filtered_lines.append([new_points_indices[lines[i,0]], new_points_indices[lines[i,1]]])
        filtered_lines = np.array(filtered_lines)
        mesh.cells[j] = meshio.CellBlock("line", np.vstack((lines, filtered_lines)))
        N = np.shape(mesh.cells[j].data)[0]
        mesh.cell_data['gmsh:physical'][j] = np.array([ids[mesh.cell_data['gmsh:physical'][j][0]]]*N)
        mesh.cell_data['gmsh:geometrical'][j] = np.array([mesh.cell_data['gmsh:geometrical'][j][0]]*N)


print("new_points", np.shape(new_points))
meshio.write(mesh_file[:-3]+"xdmf", mesh)  # XDMF for FEniCS

print("Full mesh generated successfully!")


'''
This part read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf 
'''

print( output_dir + "mesh.xdmf" )
mesh_from_file = meshio.read( output_dir + "/mesh.xdmf" )

line_mesh = msh.create_mesh( mesh_from_file, "line", prune_z=True )
meshio.write( output_dir + "/line_mesh.xdmf", line_mesh )

triangle_mesh = msh.create_mesh( mesh_from_file, "triangle", prune_z=True )
meshio.write( output_dir + "/triangle_mesh.xdmf", triangle_mesh )

print("Mesh generated and saved to ", output_dir)
