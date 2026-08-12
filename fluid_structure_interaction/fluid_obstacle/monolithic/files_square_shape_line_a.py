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
    # Eq. (108)
    '<<(\sigma^shape_{ij} G^n_{kj} nu_k - (sigma^square_{ij} G^n_{kj} nu_k  + 1/|F^n| f^i)) (\sigma^shape_{il} G^n_{ml} nu_m - (sigma^square_{il} G^n_{ml} nu_m  + 1/|F^n| f^i))>>_{partial Omega circle}', \
    # Eq. (112)
    '<<|v^n - v_lrb|^2>>_{partial Omega lrb}', \
    # Eq. (113)
    '<<||F^n| varsigma^square_{alpha beta} G^n_{gamma beta} nu_gamma - (textrm{t}^n_alpha)|^2>>_{partial Omega t}', \
    # Eq. (122)
    '<<[v^n_i]_j [v^n_i]_j>>_{partial Omega circle}' , \
    # Eq. (114)
    '<<|sigma^n - sigma_{square t}|^2>>_{partial Omega t}' ,\
    # Eq. (95)
    '<<|u^n|^2>>_{partial Omega square}',\
    # Eq. (96) projected on n_cur
    '<<((u^n - u^{n-1} - v^n dt) . n_cur)^2>>_{partial Omega shape}',\
    # Eq. (97)
    '<<|dot{u}^n|^2>>_{partial Omega square}',\
    # Eq. (98) projected on n_cur
    '<<((dot{u}^n - v^n) . n_cur)^2>>_{partial Omega shape}',\
    # Eq. (99)
    '<<[u^n_i]_j [u^n_i]_j>>_{partial Omega circle}',\
    # Eq. (100)
    '<<[dot{u}^n_i]_j [dot{u}^n_i]_j>>_{partial Omega circle}',\
    # Eq. (124)
    '<<G^n_{gamma alpha} nu_gamma (-D G^n_{beta alpha} partial c^n / partial y_beta)>>_{partial Omega square}',\
    # Eq. (125)
    '<<|F^n| G^n_{gamma alpha} nu_gamma (-D G^n_{beta alpha} partial c^n / partial y_beta ) - kappa >>_{partial Omega shape}'
    ]
writer_bcs = csv.DictWriter(csvfile_bcs, fieldnames=fieldnames_bcs)
writer_bcs.writeheader()


# 3 IC file
# create the path for the csv file if it does not exist
filepath_ics = os.path.join(rarg.args.output_directory, 'ics.csv')
os.makedirs(os.path.dirname(filepath_ics), exist_ok=True)

csvfile_ics = open(filepath_ics, 'a', newline='')
fieldnames_ics = [ \
    'step',\
    # continuity of v^n in \Omega shape
    '<<[v^n_i]_j [v^n_i]_j>>_{Omega shape}',\
    # continuity of v^n in \Omega square
    '<<[v^n_i]_j [v^n_i]_j>>_{Omega square}',\
    # continuity of sigma^n in \Omega shape
    '<<[sigma^n]_i [sigma^n]_i>>_{Omega shape}',\
    # continuity of sigma^n in \Omega square
    '<<[sigma^n]_i [sigma^n]_i>>_{Omega square}',\
    # continuity of u^n in \Omega shape
    '<<[u^n_i]_j [u^n_i]_j>>_{Omega shape}',\
    # continuity of u^n in \Omega square
    '<<[u^n_i]_j [u^n_i]_j>>_{Omega square}',\
    # continuity of dot{u}^n in \Omega shape
    '<<[dot{u}^n_i]_j [dot{u}^n_i]_j>>_{Omega shape}',\
    # continuity of dot{u}^n in \Omega square
    '<<[dot{u}^n_i]_j [dot{u}^n_i]_j>>_{Omega square}',\
    # continuity of c^n in \Omega square
    '<<[c^n]_i [c^n]_i>>_{Omega square}'
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
    '<v^n>_{Omega shape}',
    '<|f_fluid|>_{partial Omega shape}',
    '<|f_shape|>_{partial Omega shape}'
    ]
writer_data = csv.DictWriter(csvfile_data, fieldnames=fieldnames_data)
writer_data.writeheader()


