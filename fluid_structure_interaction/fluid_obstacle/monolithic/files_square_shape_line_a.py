import csv
from fenics import *
import os

import runtime_arguments as rarg

# 1  XDMF files


# 2 BC file
filepath_bcs = os.path.join(rarg.args.output_directory, 'bcs.csv')
os.makedirs(os.path.dirname(filepath_bcs), exist_ok=True)

csvfile_bcs = open(filepath_bcs, 'a', newline='')
fieldnames_bcs = [ \
    'step', \
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
    'step'
    ]
writer_data = csv.DictWriter(csvfile_data, fieldnames=fieldnames_data)
writer_data.writeheader()


