"""
Logic of association priority with situation.
"""

def cond_algorithm(left_side:float, right_side:float, front_side:float, back_side:float, 
                   x_pos:float, x_velocity:float, y_pos:float, y_velocity:float, distance:float) -> tuple:
    """
    Conditional algorithm of detecting car accident with pedestrian.

    Args:
        left_side (float): The intersection point of the pedestrian's speed vector with the left side of the vehicle.
        right_side (float): The intersection point of the pedestrian's speed vector with the right side of the vehicle.
        front_side (float): The intersection point of the pedestrian's speed vector with the front side of the vehicle.
        back_side (float): The intersection point of the pedestrian's speed vector with the back side of the vehicle.
        x_pos (float): Value of X coordinate of pedestrian in cartesian coordinate system.
        x_velocity (float): Value of X coordinate velocity.
        y_pos (float): Value of Y coordinate of pedestrian in cartesian coordinate system.
        y_velocity (float): Value of Y coordinate velocity.
        distance (float): Distance from car to pedestrian.

    Returns:
        tuple: priority and distance to the object 
    """

    if (-2 < left_side < 5) or (-2 < right_side < 5) or (-3 < front_side < 3) or (-3 < back_side < 3):
        if x_velocity > 0 :
            if distance < 5 :
                return 3, distance
            else:
                return 2, distance
        else:
            if 0 < x_pos < 30:
                if -7 < y_pos < 7:
                    if x_velocity > -2:
                        return 2, distance
                    elif x_velocity > -4:
                        return 1, distance
                    else:
                        return 0, distance
                else:
                    return 3, distance
            else: 
                    return 3, distance
    else:
        if 0 < x_pos < 20:
            if -2 < y_pos < 2:
                return 2, distance
            else:
                return 3, distance
        else:
            return 3, distance
