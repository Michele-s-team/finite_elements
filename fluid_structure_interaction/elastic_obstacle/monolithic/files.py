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



# 3 BC file
filepath_bcs = os.path.join(rarg.args.output_directory, 'bcs.csv')
os.makedirs(os.path.dirname(filepath_bcs), exist_ok=True)

csvfile_bcs = open(filepath_bcs, 'a', newline='')
fieldnames_bcs = [ \
    'step', \
    '<<|v^n - v_l|^2>>_{partial Omega l}', \
    '<<|v^n - v_tb|^2>>_{partial Omega tb}',\
    '<<|v^{n square} - average(u_dot_n)|^2>>_{partial Omega ellipse}',\
    '<<varsigma_{i 1} varsigma_{i 1}>>_{partial Omega r}',\
    '<<varsigma^2>>_{partial Omega r}',\
    '<<(nu_j P_{ij} - vasigma_{ij} |F| G_{kj} nu_k) (nu_j P_{il} - vasigma_{il} |F| G_{ml} nu_m)>>_{partial Omega circle}', \
    '<<|u^n|^2>>_{partial Omega square}', \
    '<<[u^n_i]_j [u^n_i]_j>>_{partial Omega ellipse}',\
    '<<|\dot{u}^n|^2>>_{partial Omega square}', \
    '<<[\dot{u}^n_i]_j [\dot{u}^n_i]_j>>_{partial Omega ellipse}'
    ]
writer_bcs = csv.DictWriter(csvfile_bcs, fieldnames=fieldnames_bcs)
writer_bcs.writeheader()


# 4 IC file
# create the path for the csv file if it does not exist
filepath_ics = os.path.join(rarg.args.output_directory, 'ics.csv')
os.makedirs(os.path.dirname(filepath_ics), exist_ok=True)

csvfile_ics = open(filepath_ics, 'a', newline='')
fieldnames_ics = [ \
    'step',
    '<<[v^n_i]_j [v^n_i]_j>>_{partial Omega square I}',
    '<<[varsigma]_i [varsigma]_i>>_{partial Omega square I}',
    '<<[u^n_i]_j [u^n_i]_j>>_{partial Omega circle I}',
    '<<[u^n_i]_j [u^n_i]_j>>_{partial Omega square I}',
    '<<[\dot{u}^n_i]_j [\dot{u}^n_i]_j>>_{partial Omega circle I}',
    '<<[\dot{u}^n_i]_j [\dot{u}^n_i]_j>>_{partial Omega square I}'
    ]
writer_ics = csv.DictWriter(csvfile_ics, fieldnames=fieldnames_ics)
writer_ics.writeheader()

