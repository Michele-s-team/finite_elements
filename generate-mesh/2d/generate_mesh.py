'''
To create an instance use "name = generate_mesh( ... )"
To generate a specific mesh (example square mesh) ues "name.generate_square_mesh(...) "
To save the mesh to a XDMF file use "export_mesh_as_xdmf(...)"
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

    def generate_square_symmetric_mesh(self, L, h, r, c_r = [0.0, 0.0, 0.0]):
        N = int(np.round(1/self.resolution))
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
        self.model.add_physical( [channel_lines[4],channel_lines[0]], "b" )
        self.model.add_physical( channel_lines[5:], "c" )

        self.geometry.generate_mesh( dim=2 )
        gmsh.write( self.mesh_file)

        #msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

        gmsh.clear()
        self.geometry.__exit__()



        '''
        mesh = geom.generate_mesh()

        points = mesh.points
        cells = mesh.cells_dict["triangle"]  # Adjust for other element types

        # Mirror points
        mirrored_points = points.copy()
        mirrored_points[:, axis] *= -1  # Reflect along the specified axis

        # Merge original and mirrored points
        combined_points = np.vstack((points, mirrored_points))

        # Adjust cell indices for mirrored mesh
        mirrored_cells = cells + len(points)  # Shift indices
        combined_cells = np.vstack((cells, mirrored_cells))

        return meshio.Mesh(points=combined_points, cells={"triangle": combined_cells})
        '''

    def generate_square_mesh_symm(self, L, h, r, c_r = [0.0, 0.0, 0.0]):
        N = int(np.round(1/self.resolution))
        half_rectangle_points = [self.model.add_point( (L/2, 0, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (L/2, h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (-L/2, h/2, 0), mesh_size=resolution*(min(L,h)/r) ),
                    self.model.add_point( (-L/2, 0, 0), mesh_size=resolution*(min(L,h)/r) ),
                    ]
        
        #half_circle_t_points = [self.model.add_point( (-r*np.cos(np.pi*i/N), r*np.sin(np.pi*i/N), 0), mesh_size=resolution ) for i in range(N+1)]
        half_circle_t_points = [self.model.add_point( (r*np.cos(np.pi*i/N), -r*np.sin(np.pi*i/N), 0), mesh_size=resolution ) for i in range(N+1)]
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
        self.model.add_physical( [channel_lines[4],channel_lines[0]], "b" )
        self.model.add_physical( channel_lines[5:], "c" )

        self.geometry.generate_mesh( dim=2 )
        gmsh.write( self.mesh_file)

        #msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

        gmsh.clear()
        self.geometry.__exit__()

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



    def export_mesh_as_xdmf(self):
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
    L = 6
    h = 6
    c_r = [0, 0, 0]
    c_R = [0, 0, 0]

    gmesh = generate_mesh(resolution, args.output_dir)
    gmesh.generate_square_symmetric_mesh(L,h,r)
    gmesh.export_mesh_as_xdmf()



