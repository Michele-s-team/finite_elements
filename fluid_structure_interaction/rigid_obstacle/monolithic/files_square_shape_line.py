import csv
from fenics import *
import os

import runtime_arguments as rarg

# 1  XDMF files
xdmffile_v_n = XDMFFile(os.path.join(rarg.args.output_directory, "v_n.xdmf"))
xdmffile_sigma_n = XDMFFile(os.path.join(rarg.args.output_directory, "sigma_n.xdmf"))

xdmffile_u_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_n.xdmf"))
xdmffile_u_dot_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_dot_n.xdmf"))



# 2 BC file
filepath_bcs = os.path.join(rarg.args.output_directory, 'bcs.csv')
os.makedirs(os.path.dirname(filepath_bcs), exist_ok=True)

csvfile_bcs = open(filepath_bcs, 'a', newline='')
fieldnames_bcs = [ \
    'step',
    # These two equations are Eq. (59) in 'Channel flow with moving obstacle'
    '<<|\\textrm{v}^n - v_l|^2>>_{partial Omega l}', \
    '<<|\\textrm{v}^n - v_tb|^2>>_{partial Omega tb}',\
    # Eq. (60) in 'Channel flow with moving obstacle'
    '<<|\\textrm{v}^n - omega^n \\partial_\\theta R (y -c)|^2>>_{partial Omega shape}',\
    # Eq. (105) in 'Channel flow with moving obstacle'
    '<<\\varsigma_n^2>>_{partial Omega r}',\
    # Eq. (106) in 'Channel flow with moving obstacle'
    '<<\\varsigma_{i 1} \\varsigma_{i 1}>>_{partial Omega r}',\
    # Eq. (99) in 'Channel flow with moving obstacle'
    '<<|u^n|^2>>_{partial Omega square}',\
    # Eq. (100) in 'Channel flow with moving obstacle'
    '<<|u^n - ((c - y) + R(\\theta_n - \\theta_ref).(y-c))|^2>>_{partial Omega shape}',\
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
    'theta_n',
    'omega_n',
    'pressure_scale',
    'mesh_quality'
    ]
writer_data = csv.DictWriter(csvfile_data, fieldnames=fieldnames_data)
writer_data.writeheader()


