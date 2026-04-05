import numpy as np

import constants.utils as const


'''
check whether a point lies on the line between two other points
Input values: 
    * Mandatory:
        - 'p': the point [p_x, p_y, p_z]
        - 'a', 'b': the two points defining the line [a_x, a_y, a_z], [b_x, b_y, b_z]
    * Optional: 
        - 'epsilon': the tolerance to check that p sits on the infinite line that goes through a and b
        
Return values: 
    - True if the  points lies on the line, False otherwise 

'''
def between_points(p, a, b,
                   epsilon = const.epsilon):

    dr = np.subtract(b, a)
    dr_norm = np.linalg.norm(dr)

    dr_normalized = dr/dr_norm

    p_minus_a = np.subtract(p, a)

    # check that p lies on the inifinite line that goes through a and b, and that, on that it lies on the portion of that line that is between a and b
    if (np.linalg.norm(np.subtract(p_minus_a, np.dot(p_minus_a, dr_normalized) * dr_normalized)) < epsilon) and (0 <= np.dot(p_minus_a, dr_normalized) <= dr_norm):

        return True
    
    else: 
       
        return False