from matplotlib.path import Path
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
    
'''
check whether a point is in the region delimited by a polygon
Input values: 
    - 'x': [X, Y] the coordinates of the point
    - 'polygon_coordinates': [[p0x, p0y], [p1x, p1y], ..., ] the coordinates of the points of the polygon. The last point of polygon_coordinates does not coincide with the first point, i.e., len(polygon_coordinates) = [number of vertices of the polygon]
Return values: 
    - 'True' ('False') if x belongs (does not belong to the polygon)
'''

def in_polygon(x, polygon_coordinates):

    return Path(polygon_coordinates).contains_points([x])[0]


'''
compute the aspect ratio of a polygon in two dimensions
Input values: 
    - `polygon_coordinates`: [[p_0_x, p_0_y], [p_1_x, p_1_y], ...] the coordinates of the points of the polygon (the last point is not equal to the first)

Return values: 
    - the aspect ratio ([maximal y coordinate] - [minimal y coordinate])/([maximal x coordinate] - [minimal x coordinate])
'''

def aspect_ratio(polygon_coordinates):

    coord_x = [polygon_coordinates[i, 0] for i in range(len(polygon_coordinates))]
    coord_y = [polygon_coordinates[i, 1] for i in range(len(polygon_coordinates))]

    return (np.max(coord_y) - np.min(coord_y))/(np.max(coord_x) - np.min(coord_x))