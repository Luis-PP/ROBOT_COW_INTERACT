"""
Author: Luis Ponce Pacheco
Contact: luis.poncepacheco@wur.nl
PSG, ABE group.
"""

import math as m
import numpy as np
import random


def get_line(p1, p2, parts):
    return zip(
        np.linspace(p1[0], p2[0], parts, endpoint=False),
        np.linspace(p1[1], p2[1], parts, endpoint=False),
    )


def is_close(own, target, dist):
    own_x, own_y = own
    t_x, t_y = target
    if m.isclose(own_x, t_x, abs_tol=dist) and m.isclose(own_y, t_y, abs_tol=dist):
        return True
    else:
        return False


def is_out(own, target):
    own_x, own_y = own
    t_x, t_y = target.pos
    x_offset, y_offset = target.offset
    t_x0 = t_x - x_offset
    t_x1 = t_x + x_offset
    t_y0 = t_y - y_offset
    t_y1 = t_y + y_offset
    if own_x > t_x0 and own_x < t_x1 and own_y > t_y0 and own_y < t_y1:
        return False
    else:
        return True


def opposite_direction(own, target, width, height):
    own_x, own_y = own
    t_x, t_y = target
    dx = t_x - own_x
    dy = t_y - own_y

    if dx >= 0:
        x_opposite = random.uniform(0, own_x)
    else:
        x_opposite = random.uniform(own_x, width)

    if dy >= 0:
        y_opposite = random.uniform(0, own_y)
    else:
        y_opposite = random.uniform(own_y, height)

    return (x_opposite, y_opposite)
