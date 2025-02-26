import numpy as np


#return the radian angle of vector r by taking into account its quadrant
def atan_quad(r):
    if (r[0] > 0):
        angle = np.arctan( r[1] / r[0] )
    else:
        angle = np.pi + np.arctan( r[1] / r[0] )

    return angle - 2* np.pi * np.floor(angle/(2*np.pi))