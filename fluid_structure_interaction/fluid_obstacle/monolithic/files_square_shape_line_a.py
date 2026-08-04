import csv
from fenics import *
import os

import runtime_arguments as rarg

# 1  XDMF files
xdmffile_v_n = XDMFFile(os.path.join(rarg.args.output_directory, "v_n.xdmf"))
xdmffile_sigma_n = XDMFFile(os.path.join(rarg.args.output_directory, "sigma_n.xdmf"))

xdmffile_u_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_n.xdmf"))
xdmffile_u_dot_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_dot_n.xdmf"))

xdmffile_c_n = XDMFFile(os.path.join(rarg.args.output_directory, "c_n.xdmf"))

xdmffile_mu_n = XDMFFile(os.path.join(rarg.args.output_directory, "mu_n.xdmf"))
xdmffile_grad_u_n = XDMFFile(os.path.join(rarg.args.output_directory, "grad_u_n.xdmf"))

xdmffile_u_0 = XDMFFile(os.path.join(rarg.args.output_directory, "u_0.xdmf"))


# 2 BC file
filepath_bcs = os.path.join(rarg.args.output_directory, 'bcs.csv')
os.makedirs(os.path.dirname(filepath_bcs), exist_ok=True)

csvfile_bcs = open(filepath_bcs, 'a', newline='')
fieldnames_bcs = [ \
    'step', \
    # Eq. (112)
    '<<|v^n - v_lrb|^2>>_{partial Omega lrb}', \
    # Eq. (113)
    '<<||F^n| varsigma^square_{alpha beta} G^n_{gamma beta} nu_gamma - (textrm{t}^n_alpha)|^2>>_{partial Omega t}', \
    # Eq. (122)
    '<<[v^n_i]_j [v^n_i]_j>>_{partial Omega circle}'
      ]
writer_bcs = csv.DictWriter(csvfile_bcs, fieldnames=fieldnames_bcs)
writer_bcs.writeheader()


# 3 IC file
# create the path for the csv file if it does not exist
filepath_ics = os.path.join(rarg.args.output_directory, 'ics.csv')
os.makedirs(os.path.dirname(filepath_ics), exist_ok=True)

csvfile_ics = open(filepath_ics, 'a', newline='')
fieldnames_ics = [ \
    'step'
     ]
writer_ics = csv.DictWriter(csvfile_ics, fieldnames=fieldnames_ics)
writer_ics.writeheader()


# 4 data file
filepath_data = os.path.join(rarg.args.output_directory, 'data.csv')
os.makedirs(os.path.dirname(filepath_data), exist_ok=True)

csvfile_data = open(filepath_data, 'a', newline='')
fieldnames_data = [ \
    'step',
    '<u_n_y>_shape',
    '<y>_shape',
    'shape_volume',
    'mesh_quality',
    '<v^n>_{Omega circle}'
    ]
writer_data = csv.DictWriter(csvfile_data, fieldnames=fieldnames_data)
writer_data.writeheader()


