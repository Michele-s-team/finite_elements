import csv
from fenics import *
import os

import runtime_arguments as rarg

#1.   XDMF files
# 1) membrane problem
xdmffile_v_bar = XDMFFile( os.path.join( rarg.args.output_directory,  'v_bar.xdmf') )
xdmffile_w_bar = XDMFFile( os.path.join( rarg.args.output_directory,  'w_bar.xdmf') )
xdmffile_v_n = XDMFFile( os.path.join( rarg.args.output_directory,  'v_n.xdmf') )
xdmffile_w_n = XDMFFile( os.path.join( rarg.args.output_directory,  'w_n.xdmf') )
xdmffile_phi = XDMFFile( os.path.join( rarg.args.output_directory,  'phi.xdmf') )
xdmffile_sigma_n_12 = XDMFFile( os.path.join( rarg.args.output_directory,  'sigma_n_12.xdmf') )
xdmffile_u_n_12 = XDMFFile( os.path.join( rarg.args.output_directory,  'X_n_12.xdmf') )
xdmffile_nu_n_12 = XDMFFile( os.path.join( rarg.args.output_directory,  'nu_n_12.xdmf') )
xdmffile_psi_n_12 = XDMFFile( os.path.join( rarg.args.output_directory,  'psi_n_12.xdmf') )
xdmffile_mu_n_12 = XDMFFile( os.path.join( rarg.args.output_directory,  'mu_n_12.xdmf') )

# 2) mesh problem
xdmffile_u_n = XDMFFile( os.path.join(rarg.args.output_directory , 'u_n.xdmf') )
xdmffile_u_dot_n = XDMFFile( os.path.join(rarg.args.output_directory , 'u_dot_n.xdmf') )

# 3) fluid problem 
xdmffile_v_fl_n = XDMFFile( os.path.join(rarg.args.output_directory , 'v_fl_n.xdmf') )
xdmffile_v_fl_bar = XDMFFile( os.path.join(rarg.args.output_directory , 'v_bar_fl.xdmf') )
xdmffile_sigma_fl = XDMFFile( os.path.join(rarg.args.output_directory , 'sigma_fl_n_12.xdmf') )
xdmffile_phi_fl = XDMFFile( os.path.join(rarg.args.output_directory , 'phi_fl.xdmf') )

# 2. BC file


# create the path for the csv file if it does not exist
filepath_bcs = os.path.join(rarg.args.output_directory + 'bcs.csv')
os.makedirs(os.path.dirname(filepath_bcs), exist_ok=True)

csvfile_bcs = open(filepath_bcs, 'a', newline='')
fieldnames_bcs = [ \
    'step', \

    # 2.1 membrane problem
    '<<|v_bar - g|^2>>_{ds_{~ l}}',
    '<<|v_bar - g|^2>>_{ds_{~ r}}',
    '<<w_bar^2>>_{ds_{~ l}}',
    '<<phi^2>>_{ds_{~ l}}',
    '<<|U^{n-1/2}|^2>>_{ds_{~ l}}',
    '<<(U^{n-1/2, 1})^2>>_{ds_{~ r}}',
    '<<(n^{n-1/2, i} \\Nabla^{n-1/2}_i phi)^2>>_{ds_{~}}',
    '<<(\\Nabla^{n-1/2}_i w_bar)^2>>_{ds_{~ r}}',
    '<<(\\partial U^{n-1/2, 2} / \\partial x^i)^2>>_{ds_{~ r}}',
    
    # 2.2 mesh problem
    # u problem
    '<<|u^n|^2>>_{ds_{square lb}}',
    '<<({u^n}^1)^2>>_{ds_{square r}}',
    '<<|u^n - U^{n-1/2}|^2>>_{ds_{square t}}',
    '<<(\\partial u^{n, 2} / \\partial y^1)^2>>_{ds_{square r}}',
    # u_dot problem
    '<<|u_dot^n|^2>>_{ds_{square lb}}',
    '<<({u_dot^n}^1)^2>>_{ds_{square r}}',
    '<<|u_dot^n - U_dot^{n-1/2}|^2>>_{ds_{square t}}',
    '<<(\\partial u_dot^{n, 2} / \\partial y^1)^2>>_{ds_{square r}}',

    # 2.3 fluid problem
    '<<|v_bar_fl - h|^2>>_{ds_{square b}}',
    '<<|v_bar_fl|^2>>_{ds_{square l}}',
    '<<|v_bar_fl_0|^2>>_{ds_{square r}}',
    '<<|v_bar_fl - u_dot_n|^2>>_{ds_{square t}}',
    '<<|phi_fl|^2>>_{ds_{square b}}',
    '<<(G^{n-1}_{alpha 1 \partial V_{FL}^2 / \partial y^\alpha})^2>>_{ds_{square r}}',
    '<<(nu_gamma G^{n-1}_{gamma alpha} G^{n-1}_{beta alpha} \\partial \\phi_{FL} / \\partial y^\\beta)^2>>_{ds_{square lt}}',
    '<<(G^{n-1}_{beta 1} \\partial phi_{FL} / \\partial y_beta )^2>>_{ds_{square r}}'   
    ]

writer_bcs = csv.DictWriter(csvfile_bcs, fieldnames=fieldnames_bcs)
writer_bcs.writeheader()