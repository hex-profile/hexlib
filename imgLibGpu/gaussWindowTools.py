from numpy import *
import numpy as np

#================================================================
#
# gauss
#
#================================================================

def gauss(x, s): 
    return exp(-square(x) / square(s) / 2) / sqrt(2 * pi) / s

#================================================================
#
# computeFilterStartIndex
#
#================================================================

def computeFilterStartIndex(filterCenter, taps):
    startPos = filterCenter - 0.5 * (taps - 1) # taps-1 is a must!
    return floor(startPos).astype(int)

#================================================================
#
# gaussSum
#
#================================================================

def gaussSum(filterCenter, taps, sigma, indexCorrection = 0):
    startIndex = computeFilterStartIndex(filterCenter, taps) + indexCorrection
    tapSamples = arange(startIndex, startIndex + taps) + 0.5
    # print(tapSamples)
    g = gauss(tapSamples - filterCenter, sigma)
    # print(g)
    return sum(g)

#================================================================
#
# gaussCoverage
#
#================================================================

def gaussCoverage(filterCenter, taps, sigma):
    return gaussSum(filterCenter, taps, sigma) / gaussSum(filterCenter, taps + 256, sigma)

#================================================================
#
# movingCoverage
#
#================================================================

def movingCoverage(taps, sigma):
    
    filterCenters = linspace(0, 1, num=101)

    worstCoverage = 1

    for filterCenter in filterCenters:
        worstCoverage = min(worstCoverage, gaussCoverage(filterCenter, taps, sigma))

    return worstCoverage

#================================================================
#
# main
#
#================================================================

if __name__ == '__main__':

    sigma = 0.001

    desiredCoverage = 255.0/256 # 8-bit accuracy

    for taps in range(1, 16 + 1):

        goodSigma = sigma

        while (movingCoverage(taps, sigma) >= desiredCoverage):
            goodSigma = sigma
            sigma += 0.001

        print('%.3ff, // %d taps' % (goodSigma, taps))
