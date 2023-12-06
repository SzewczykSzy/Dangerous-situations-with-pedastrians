import numpy as np
from functions.conditional_algorithm import cond_algorithm
from filterpy.kalman import KalmanFilter
import math

EPS = np.finfo(float).eps
DT = 0.1


def linear_func_coefficients(p1:tuple, p2:tuple) -> tuple:
    """
    Funkcja zwracająca współczynniki prostej przechodzącej przez 2 punkty p1 i p2
    """
    if (p2[0] - p1[0]) == 0:
        a = (p2[1]-p1[1])/(EPS)
    else:
        a = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p1[1] - a*p1[0]
    return a, b


def danger_sit(out_x:KalmanFilter, out_y:KalmanFilter, id:int) -> tuple:
    x_pos = out_x.x[0][0]
    y_pos = out_y.x[0][0]
    x_v = out_x.x[1][0]
    y_v = out_y.x[1][0]

    distance = math.sqrt(x_pos**2 + y_pos**2) - 1.5 # -1.5 because of the fact, that LiDAR is placed in the center of a car

    p1 = (x_pos, y_pos)
    p2 = (x_pos + x_v*DT, y_pos + y_v*DT)

    a, b = linear_func_coefficients(p1, p2)

    if a == 0:
        a = EPS

    l_s = (2-b)/a       # lower than 5 and greater than -2 
    r_s = (-2-b)/a      #           -||-

    f_s =  a*5 + b      # lower than 2 and greater than -2
    b_s = a*(-2) + b    #           -||- 

    return cond_algorithm(left_side=l_s, right_side=r_s, front_side=f_s, back_side=b_s, 
                          x_pos=x_pos, x_velocity=x_v, y_pos=y_pos, y_velocity=y_v, distance=distance)