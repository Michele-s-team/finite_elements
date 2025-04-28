'''
this code performs a test of mlutiple parts of the code by comparing the solution in csv files across two commits

run with

python3 test.py [sha of commit_a] [sha of commit_b]
Example
python3 test.py 7e670cf2a5ba005ab77202c124d691eaa5bc17ea 0119597915ccfedf8560e5092c2cda8ae74ce152
'''

import colorama as col
import os
import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import runtime_arguments as rarg
import unit_test as utes

commit_a = rarg.args.commit_a
commit_b = rarg.args.commit_b

root_path = io.add_trailing_slash('/home/fenics/shared')

print(f'commit_a = {commit_a}')
print(f'commit_b = {commit_b}')

utes.checkout(commit_a)
utes.checkout(commit_b)


utes.checkout('unit_test')


