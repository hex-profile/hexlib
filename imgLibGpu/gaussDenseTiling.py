from numpy import *
import numpy as np
import math

#================================================================
#
# gauss
#
#================================================================

def gauss(x, s): 
    return exp(-square(x) / square(s) / 2) / sqrt(2 * pi) / s

#================================================================
#
# main
#
#================================================================

if __name__ == '__main__':

    s = 0.55

    #----------------------------------------------------------------
    #
    # 1D
    #
    #----------------------------------------------------------------

    x = linspace(-4, +4, 8*100)

    gaussSum = zeros_like(x)

    for k in range(-8, +8 + 1):
        gaussSum += gauss(x - k, s)

    err = gaussSum - 1

    avgErr = std(err)
    worstErr = amax(abs(err))
    print('1D case: avg %.1f bits, worst %.1f bits\n' % (-math.log2(avgErr), -math.log2(worstErr)))

    #----------------------------------------------------------------
    #
    # 2D
    #
    #----------------------------------------------------------------

    x = matrix(x)
    gaussSum = zeros_like(x.T @ x)

    for kx in range(-8, +8 + 1):
        for ky in range(-8, +8 + 1):
            gaussSum += gauss(x - kx, s).T @ gauss(x - ky, s)

    err = gaussSum - 1

    avgErr = std(err)
    worstErr = amax(abs(err))

    print('2D case: avg %.1f bits, worst %.1f bits\n' % (-math.log2(avgErr), -math.log2(worstErr)))
