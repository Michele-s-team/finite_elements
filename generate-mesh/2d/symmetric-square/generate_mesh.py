'''
If you want to use call this module from another file you need the following:
    To create an instance use "name = generate_mesh( ... )"
    To generate a specific mesh (example square mesh) use "name.generate_square_mesh(...) "
    To save the mesh to a XDMF file use "export_mesh_as_xdmf(...)"
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
module_path = '/home/tanos/Thesis/Fenics-files-for-thesis/modules/'
sys.path.append( module_path )
from mesh import mesh as msh

msh = msh()

class generate_mesh:
    def __init__(self, resolution, output_dir):
        # Initialize empty geometry using the build in kernel in GMSH
        self.geometry = pygmsh.geo.Geometry()
        self.model = self.geometry.__enter__()
        self.output_dir = output_dir
        self.resolution = resolution
        self.mesh_file = output_dir + "/mesh.msh"

    def print_param(self):
        for el in self.param:
            print(el, "=", self.param[el])

    def generate_half_square_mesh(self, L, h, r, c_r = [0.0, 0.0, 0.0]):
        N = int(np.round(np.pi/(self.resolution)))
        half_rectangle_points = [self.model.add_point( (L/2, 0, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (L/2, h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (-L/2, h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (-L/2, 0, 0), mesh_size=resolution*(min(L,h)/r) ),
                    ]
        
        half_circle_points = [self.model.add_point( (-r*np.cos(np.pi*i/N), r*np.sin(np.pi*i/N), 0), mesh_size=resolution ) for i in range(N+1)]
        my_points = half_rectangle_points + half_circle_points
        channel_lines = [self.model.add_line( my_points[i], my_points[i + 1] )
                        for i in range( -1, len( my_points ) - 1 )]
        
        channel_loop = self.model.add_curve_loop( channel_lines )
        plane_surface = self.model.add_plane_surface( channel_loop )

        self.model.synchronize()

        self.model.add_physical( [plane_surface], "Volume" )
        self.model.add_physical( [channel_lines[1]], "r" )
        self.model.add_physical( [channel_lines[3]], "l" )
        self.model.add_physical( [channel_lines[2]], "t" )
        #self.model.add_physical( [channel_lines[4],channel_lines[0]], "b" )
        self.model.add_physical( channel_lines[5:], "c" )

        self.geometry.generate_mesh( dim=2 )
        gmsh.write( self.mesh_file)

        #msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

        gmsh.clear()
        self.geometry.__exit__()
        
    # This function is used to mirror the points of the mesh and change the points data accordingly
    def mirror_points(self, points, point_data, ids):
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

        
    # This function is used to generate a square mesh exactly simmetric with respect to the x-axis
    def generate_square_symmetric_mesh(self, L, h, r, c_r = [0.0, 0.0, 0.0]):
        # The new mesh inherits the ids (physical id used for measure definito) of the original one, except for the new physical objects that are generated from reflection (e.g. the b line)
        # in particular the rule 4:5 implies that the lines that in the original mesh where in the physical group 4 (top lines), when reflected, they will be assigned the id 5 (used to define measure in the bottom line)
        ids = [0, 1, 2, 3, 5, 6] #{1:1, 2:2, 3:3, 4:5, 5:6} 
        self.generate_half_square_mesh(L, h, r, c_r)
        # Load the half-mesh
        mesh = meshio.read(self.mesh_file)
        print("original points", np.shape(mesh.points))
        # Mirror points across X=0
        new_points, new_points_indices, new_point_data = self.mirror_points(mesh.points, mesh.point_data, ids)

        # Adjust connectivity (ensure indices match the new points array)
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
        meshio.write(self.mesh_file[:-3]+"xdmf", mesh)  # XDMF for FEniCS

        print("Full mesh generated successfully!")

    def generate_square_mesh(self, L, h, r, c_r = [0.0, 0.0, 0.0]):
        self.L = L
        self.h = h
        self.r = r
        self.c_r = c_r
        self.param = {"L":L, "h":h, "r":r, "c_r":c_r}

        self.print_param()

        my_points = [self.model.add_point( (L/2, h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (-L/2, h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (-L/2, -h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (L/2, -h/2, 0), mesh_size=resolution*(min(L,h)/r) )]

        # Add lines between all points creating the rectangle
        channel_lines = [self.model.add_line( my_points[i], my_points[i + 1] )
                        for i in range( -1, len( my_points ) - 1 )]

        channel_loop = self.model.add_curve_loop( channel_lines )

        circle_r = self.model.add_circle( c_r, r, mesh_size=resolution )

        plane_surface = self.model.add_plane_surface( channel_loop, holes=[circle_r.curve_loop] )

        self.model.synchronize()

        self.model.add_physical( [plane_surface], "Volume" )
        self.model.add_physical( [channel_lines[0]], "i" )
        self.model.add_physical( [channel_lines[2]], "o" )
        self.model.add_physical( [channel_lines[3]], "t" )
        self.model.add_physical( [channel_lines[1]], "b" )
        self.model.add_physical( circle_r.curve_loop.curves, "c" )

        self.geometry.generate_mesh( dim=2 )
        gmsh.write( self.mesh_file)

        #msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

        gmsh.clear()
        self.geometry.__exit__()


    def generate_ring_mesh(self, r, R, c_r=[0, 0, 0], c_R=[0, 0, 0]):
        self.r = r
        self.R = R 
        self.c_r = c_r
        self.c_R = c_R
        self.param = {"r":r, "R":R, "c_r":c_r, "c_R":c_R}

        self.print_param()

        circle_r = self.model.add_circle(c_r, r, mesh_size=resolution)
        circle_R = self.model.add_circle(c_R, R, mesh_size=resolution*(R/r))

        plane_surface = self.model.add_plane_surface(circle_R.curve_loop, holes=[circle_r.curve_loop])

        self.model.synchronize()
        self.model.add_physical([plane_surface], "Volume")

        #I will read this tagged element with `ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)`
        self.model.add_physical(circle_r.curve_loop.curves, "Circle r")
        self.model.add_physical(circle_R.curve_loop.curves, "Circle R")
        
        self.geometry.generate_mesh(64)
        gmsh.write(self.output_dir + "/mesh.msh")
        gmsh.clear()
        self.geometry.__exit__()

    def export_mesh_as_xdmf_from_xdmf(self):
        mesh_from_file = meshio.read( self.output_dir + "/mesh.xdmf" )

        line_mesh = msh.create_mesh( mesh_from_file, "line", prune_z=True )
        meshio.write( self.output_dir + "/line_mesh.xdmf", line_mesh )

        triangle_mesh = msh.create_mesh( mesh_from_file, "triangle", prune_z=True )
        meshio.write( self.output_dir + "/triangle_mesh.xdmf", triangle_mesh )

        print("Mesh generated and saved to ", self.output_dir)

    def export_mesh_as_xdmf_from_msh(self):
        mesh_from_file = meshio.read( self.mesh_file )

        line_mesh = msh.create_mesh( mesh_from_file, "line", prune_z=True )
        meshio.write( self.output_dir + "/line_mesh.xdmf", line_mesh )

        triangle_mesh = msh.create_mesh( mesh_from_file, "triangle", prune_z=True )
        meshio.write( self.output_dir + "/triangle_mesh.xdmf", triangle_mesh )

        print("Mesh generated and saved to ", self.output_dir)


    

if __name__ == "__main__":
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

    gmesh = generate_mesh(resolution, args.output_dir)
    gmesh.generate_square_symmetric_mesh(L,h,r)
    gmesh.export_mesh_as_xdmf_from_xdmf()



