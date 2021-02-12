from numpy import *
import numpy as np
import math, sys

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

    s = 0.6

    #----------------------------------------------------------------
    #
    # 1D
    #
    #----------------------------------------------------------------

    x = linspace(-4, +4, 8*100)

    gaussSum = zeros_like(x)

    for k in range(-16, +16 + 1):
        g = gauss(x - k, s)
        gaussSum += g

    err = gaussSum - 1
    avgErr = mean(abs(err))
    worstErr = amax(abs(err))
    print('1D case: avg %.1f bits, worst %.1f bits\n' % (-math.log2(avgErr), -math.log2(worstErr)))

    #----------------------------------------------------------------
    #
    # 2D
    #
    #----------------------------------------------------------------

    if False:
        x = matrix(x)
        gaussSum = zeros_like(x.T @ x)

        for kx in range(-8, +8 + 1):
            for ky in range(-8, +8 + 1):
                gaussSum += gauss(x - kx, s).T @ gauss(x - ky, s)

        err = gaussSum - 1

        avgErr = std(err)
        worstErr = amax(abs(err))

        print('2D case: avg %.1f bits, worst %.1f bits\n' % (-math.log2(avgErr), -math.log2(worstErr)))

    #----------------------------------------------------------------
    #
    # Subsample coordinate recovery test.
    #
    #----------------------------------------------------------------

    s = 0.59

    ###

    p = arange(0, 1, 0.0001)

    sw = 0
    swv = 0

    radius = 1

    for k in range(-radius, +radius + 1 + 1):
        g = gauss(p - k, s)
        sw += g
        swv += g * k

    q = swv / sw

    meanErr = max(abs(p - q))
    print('Gauss recovery %.2f bits' % (-log2(meanErr)))

    coverageErr = max(abs(sw - 1))
    print('Gauss summation %.2f bits' % (-log2(coverageErr)))
