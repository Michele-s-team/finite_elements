import csv
from fenics import *
import os

import runtime_arguments as rarg

# 1  XDMF files
xdmffile_v_n = XDMFFile(os.path.join(rarg.args.output_directory, "v_n.xdmf"))
xdmffile_sigma_n = XDMFFile(os.path.join(rarg.args.output_directory, "sigma_n.xdmf"))

xdmffile_u_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_n.xdmf"))
xdmffile_u_dot_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_dot_n.xdmf"))

xdmffile_det_F_n = XDMFFile(os.path.join(rarg.args.output_directory, "det_F_n.xdmf"))
xdmffile_u_0 = XDMFFile(os.path.join(rarg.args.output_directory, "u_0.xdmf"))


# 2 data file
filepath_data = os.path.join(rarg.args.output_directory, 'data.csv')
os.makedirs(os.path.dirname(filepath_data), exist_ok=True)

csvfile_data = open(filepath_data, 'a', newline='')
fieldnames_data = [ \
    'step',
    '<<|u_n|^2>>_{partial Omega ellipse}',
    '<<sigma_n^2>>_{partial Omega ellipse}',
    '<<varsigma_no_pressure_{ij} varsigma_no_pressure_{ij}>>_{partial Omega ellipse}',
    'mesh_quality',
    'int_shape psi dx'
    ]
writer_data = csv.DictWriter(csvfile_data, fieldnames=fieldnames_data)
writer_data.writeheader()