

def cond_algorithm(left_side:float, right_side:float, front_side:float, back_side:float, 
                   x_pos:float, x_velocity:float, y_pos:float, y_velocity:float, distance:float) -> tuple:
    # 0 - "GO AHEAD"
    # 1 - "BE CAREFUL"
    # 2 - "SLOW DOWN"
    # 3 - "BREAK"
    # 4 - "MISTAKE"

    if (-2 < left_side < 5) or (-2 < right_side < 5) or (-3 < front_side < 3) or (-3 < back_side < 3):\
        # Człowiek się oddala
        if x_velocity > 0 :
            if distance < 5 :
                return 1, distance
            else:
                return 0, distance
        elif x_velocity < -80:
            return 4, distance
        else:
            if 0 < x_pos < 30:
                if -7 < y_pos < 7:
                    if x_velocity > -1:
                        return 1, distance
                    elif x_velocity > -3:
                        return 2, distance
                    else:
                        # print(f"BREAK: {id}\n")
                        return 3, distance
                else:
                    return 0, distance
            else: 
                return 0, distance          
    # Gdy prosta nie przecina obszaru wokół samochodu:
    else:
        if 0 < x_pos < 20:
            # Jeżeli wartość odległości współrzędnej y od obiektu jest mniejsza niż 1 m: ZWOLNIJ 
            if y_pos < 2 and y_pos > -2:
                return 1, distance
            # W przeciwnym razie: JEDŹ DALEJ
            else:
                return 0, distance
        else:
            return 0, distance
