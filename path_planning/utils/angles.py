import math


def wrap_angle(angle, lower_bound, inclusive=False):
    if inclusive:
        while angle < lower_bound:
            angle += 2 * math.pi
        while angle >= lower_bound + 2 * math.pi:
            angle -= 2 * math.pi
    else:
        while angle <= lower_bound:
            angle += 2 * math.pi
        while angle > lower_bound + 2 * math.pi:
            angle -= 2 * math.pi

    return angle


def wrap_angle_diff(angle_diff):
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    return angle_diff
