import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def kalman_filter(init:float=0, v_init:float=0) -> KalmanFilter:
    """
    Function implementing initialization of KalmanFilter

    Args:
        init (float, optional): First X/Y coordinate of pedestrian. Defaults to 0.
        v_init (float, optional): First X/Y coordinate velocity of pedestrian. Defaults to 0.

    Returns:
        KalmanFilter: KalmanFilter
    """
    my_filter = KalmanFilter(dim_x=2, dim_z=1)  # dim_x: size of the state vector
                                                # dim_z: size of the measurement vector

    dt = 0.1
    my_filter.x = np.array([[init, v_init]]).T     # x, vx

    my_filter.F = np.array([[1, dt],      # state transition matrix
                            [0, 1]])      

    my_filter.H = np.array([[1, 0]])       # Measurement function    

    my_filter.P = np.array([[1, 0],         # covariance matrix
                            [0, 1]])       

    my_filter.R = np.array([[0.1]])             # state uncertainty                  

    my_filter.Q = Q_discrete_white_noise(dim=2, dt=dt, var=10) # process uncertainty
    
    return my_filter