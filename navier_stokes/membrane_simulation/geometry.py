import numpy as np

#r, R must be the same as in generate_mesh.py
R = 1.0
r = 0.25
#these must be the same c_R, c_r as in generate_mesh.py, with the third component dropped
c_R = [0.0, 0.0]
c_r = [0.0, -0.1]


# Define norm of x
def norm(x):
    return (np.sqrt(x[0]**2 + x[1]**2))