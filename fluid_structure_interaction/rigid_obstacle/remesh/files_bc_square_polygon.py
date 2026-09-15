import csv
from fenics import *
import os
import ufl as ufl

import runtime_arguments as rarg

# 1. file for BCs

# create the path for the csv file if it does not exist
filename_bcs = os.path.join(rarg.args.output_directory, 'bcs.csv')
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile_bcs = open(filename_bcs, 'a', newline='')
fieldnames_bcs = [ \
    'step', \
    '<<(u^n_i - u_shape_i)(u^n_i - u_shape_i)>>_polygon', \
    '<<(u^n_i - u_square_i)(u^n_i - u_square_i)>>_square', \
    '<<(u_dot^n_i - u_dot_shape_i)(u_dot^n_i - u_dot_shape_i)>>_polygon', \
    '<<(u_dot^n_i - u_dot_square_i)(u_dot^n_i - u_dot_square_i)>>_square', \
    '<<(l_profile_v_bar^i - v_bar^i)(l_profile_v_bar_i - v_bar_i)>>_l', \
    '<<v_bar^i v_bar_i>>_{tb}', \
    '<<(polygon_profile_v_bar^i - v_bar^i)(v__profile_polygon - v_bar_i)>>_polygon', \
    '<<\mu G^{n-1}_{j1} \partial_j V_i>>_r', \
    '<<(G^{n-1}_{ji} nu_j G^{n-1}_{li} \partial_l phi)^2>>_{l + tb + polygon}' ,\
    '<<phi^2>>_r'
    ]

writer_bcs = csv.DictWriter(csvfile_bcs, fieldnames=fieldnames_bcs)
writer_bcs.writeheader()


# 2. file for theta omega

# create the path for the data csv file if it does not exist
theta_omega_filename = os.path.join(rarg.args.output_directory, 'theta_omega.csv')
os.makedirs(os.path.dirname(theta_omega_filename), exist_ok=True)

csvfile_theta_omega = open(theta_omega_filename, 'a', newline='')
theta_omega_fieldnames = [ "theta", "omega" ]
theta_omega_writer = csv.DictWriter(csvfile_theta_omega, fieldnames=theta_omega_fieldnames)
theta_omega_writer.writeheader()



# 3. file for data

# create the path for the csv file if it does not exist
filename_data = os.path.join(rarg.args.output_directory, 'data.csv')
os.makedirs(os.path.dirname(filename_data), exist_ok=True)

csvfile_data = open(filename_data, 'a', newline='')
fieldnames_data = [ \
    'step', \
    'mesh_quality'
    ]

writer_data = csv.DictWriter(csvfile_data, fieldnames=fieldnames_data)
writer_data.writeheader()

