from fenics import *
from dolfin import *
from mshr import *
import numpy as np

#import runtime_arguments as rarg
#import mesh as msh
#import geometry as geo
#import boundary_geometry as bgeo


class rmsh:
    def __init__(self, msh, args, bgeo, geo):
        self.args = args
        self.bgeo = bgeo
        self.msh = msh
        self.geo = geo
        #CHANGE PARAMETERS HERE
        self.r = 1.0
        self.R = 2.0
        self.c_r = [0, 0]
        self.c_R = [0, 0]

        #read the triangles
        mvc = MeshValueCollection("size_t", bgeo.mesh, bgeo.mesh.topology().dim())
        with XDMFFile((self.args.input_directory) + "/triangle_mesh.xdmf") as infile:
            infile.read(mvc, "name_to_read")
        sf = dolfin.cpp.mesh.MeshFunctionSizet(bgeo.mesh, mvc)

        #read the lines
        mvc = MeshValueCollection("size_t", bgeo.mesh, bgeo.mesh.topology().dim()-1)
        with XDMFFile((self.args.input_directory) + "/line_mesh.xdmf") as infile:
            infile.read(mvc, "name_to_read")
        mf = dolfin.cpp.mesh.MeshFunctionSizet(bgeo.mesh, mvc)


        #radius of the smallest cell in the mesh
        self.r_mesh = bgeo.mesh.hmin()

        # test for surface elements
        self.dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
        self.ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
        self.ds_R = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
        self.ds = self.ds_r + self.ds_R

        # Define boundaries and obstacle
        #CHANGE PARAMETERS HERE
        self.boundary = 'on_boundary'
        self.boundary_r = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) < (1.0 + 2.0)/2.0'
        self.boundary_R = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) > (1.0 + 2.0)/2.0'
        #CHANGE PARAMETERS HERE

    def test_mesh(self):
        #a function space used solely to define f_test_ds
        Q_test = FunctionSpace( self.bgeo.mesh, 'P', 2 )

        # f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
        f_test_ds = Function( Q_test )

        #analytical expression for a  scalar function used to test the ds
        class FunctionTestIntegralsds(UserExpression):
            def eval(self, values, x):
                c_test = [0.3, 0.76]
                r_test = 0.345
                values[0] = np.cos(self.geo.my_norm(np.subtract(x, c_test)) - r_test)**2.0
            def value_shape(self):
                return (1,)

        f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )

        self.msh.test_mesh_integral(2.90212, f_test_ds, self.dx, '\int f dx')
        self.msh.test_mesh_integral(2.77595, f_test_ds, self.ds_r, '\int f ds_r')
        self.msh.test_mesh_integral(3.67175, f_test_ds, self.ds_R, '\int f ds_R')

        