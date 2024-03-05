import math as m
import numpy as np

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
    
# def solve(bl, tr, p) :
#    if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
#       return True
#    else :
#       return False
# bottom_left = (1, 1)
# top_right = (8, 5)
# point = (5, 4)
# print(solve(bottom_left, top_right, point))
    
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
