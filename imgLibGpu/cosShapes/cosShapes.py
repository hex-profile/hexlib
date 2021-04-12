from numpy import *
import numpy as np

#================================================================
#
# cosShape
#
#================================================================

def cosShape(x): 
    t = x * (pi / 2);
    c = cos(t)
    return where(abs(x) >= 1, 0, c * c)

#================================================================
#
# main
#
#================================================================

if __name__ == '__main__':

    for n in range(1, 33 + 1):
        idx = np.arange(n)
        v = ((idx + 0.5) - 0.5 * n) / (0.5 * n)
        c = cosShape(v)
        c = c / sum(c)
        print('static devConstant float32 cosShape%d[%d] = {%s};' % (n, n, ', '.join(['%.8ff' % v for v in c])))


