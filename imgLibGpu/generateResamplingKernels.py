import cv2
import numpy as np
import argparse

#================================================================
#
# ensure
#
#================================================================

def ensure(condition, exception = AssertionError):
    if not condition:
        raise exception

class ErrorMessage(Exception):
    pass

def ensureMsg(condition, msg):
    ensure(condition, ErrorMessage(msg))

#================================================================
#
# computeFilterStartPos
#
#================================================================

def roundToInt(value):
    return np.floor(value + 0.5).astype(int)

def computeFilterStartPos(filterCenter, taps):
    return filterCenter - 0.5 * float(taps - 1) # taps - 1 is a must!

#================================================================
#
# generateFilterPack
#
# Generates a pack of aligned filters for different sampling offset.
# Used for resampling by rational number.
#
#================================================================

def generateFilterPack(name, taps, factor, normalizedVectorKernel):

    period = 2

    period = factor[0]
    scale = factor[0] / (factor[1] + 0.0)

    print('-' * 67)
    print('%s, scaling %d:%d, factor %.6f' % (name, factor[0], factor[1], scale))

    # Position of the filter center in the destintation image.
    dstCenterIdx = np.arange(period * 3)
    dstCenterPos = dstCenterIdx + 0.5
    srcCenterPos = dstCenterPos / scale

    # To recheck period
    srcRems = srcCenterPos - np.floor(srcCenterPos)
    print('Check period %d: %s' % (period, ', '.join(['%f' % v for v in srcRems])))
    print('-' * 67)

    srcCenterPos = srcCenterPos[0:period]

    # Compute optimal filter start indices.
    srcStartPos = computeFilterStartPos(srcCenterPos, taps)
    srcStartIdx = srcStartPos - 0.5
    # print(srcStartIdx)

    # Average starting index.
    # print(np.average(srcStartIdx), roundToInt(np.average(srcStartIdx)))
    avgStartIdx = roundToInt(np.average(srcStartIdx))

    # Filter starting offsets.
    startOffset = (avgStartIdx + 0.5) - srcCenterPos
    print('startOffset = ', startOffset)

    kernelScaled = lambda x, normalized: normalizedVectorKernel(x * min(scale, 1), normalized) * (1 if normalized else min(scale, 1))

    filterRange = np.arange(taps)
    largeRange = np.arange(-256, taps + 256)

    print()

    print('Filter shift = %d' % avgStartIdx)

    print()

    for p in range(period):
        sumInner = np.sum(abs(kernelScaled(filterRange + startOffset[p], False)))
        sumOuter = np.sum(abs(kernelScaled(largeRange + startOffset[p], False)))
        coverageBits = -np.log2(1 - sumInner / sumOuter)
        coeffs = kernelScaled(filterRange + startOffset[p], True)
        print('[filter%d, coverage %.2f bits]\n%s\n' % (p, coverageBits, ', '.join(['%+.8ff' % v for v in coeffs])))

#================================================================
#
# normalizeToOne
#
# Normalize for ideal color preservation when resampling.
#
#================================================================

def normalizeToOne(x, normalized):
    # print(np.sum(x))
    return x / np.sum(x) if normalized else x

#================================================================
#
# gaussSinc
# gauss
#
#================================================================

def gaussSinc(x, sigma, theta, normalized):
    return normalizeToOne(np.sin(np.pi * x / theta) / np.pi / x * np.exp(-x * x / (2 * theta * theta * sigma * sigma)), normalized)

def gauss(x, sigma, normalized):
    return normalizeToOne(np.exp(-x * x / (2 * sigma * sigma)) / np.sqrt(2 * np.pi) / sigma, normalized)

#================================================================
#
# dog
# gaussDiff
#
#================================================================

def dog(x, sigma, normalized):
    eps = 0.0001
    return (gauss(x + eps, sigma, normalized) - gauss(x - eps, sigma, normalized)) / (2 * eps)

def gaussDiff(x, sigma, normalized):
    c = 1 / (sigma * sigma) * gauss(x, sigma, normalized);
    return -x * c

#================================================================
#
# main
#
#================================================================

if __name__ == '__main__':

    #----------------------------------------------------------------
    #
    # Parse arguments
    #
    #----------------------------------------------------------------

    parser = argparse.ArgumentParser()
    # parser.add_argument('--frames', default="20")
    
    args = parser.parse_args()

    print()

    for arg in vars(args):
        print('%s = %s' % (arg, getattr(args, arg)))

    print()

    #----------------------------------------------------------------
    #
    # Gauss sinc resampling.
    #
    #----------------------------------------------------------------

    if False:

        sigma = 3
        theta = 1.3

        ###

        kernelConservative = lambda x, normalized: gaussSinc(x, sigma, theta, normalized)
        kernelBalanced = lambda x, normalized: gaussSinc(x, sigma, 1, normalized)

        #
        # 4X
        #

        generateFilterPack('Gauss-Sinc Downsample 4X', 76, [1, 4], kernelConservative)
        generateFilterPack('Gauss-Sinc Upsample 4X', 15, [4, 1], kernelBalanced)

        #
        # 2X
        #

        generateFilterPack('Gauss-Sinc Downsample 2X', 38, [1, 2], kernelConservative)
        generateFilterPack('Gauss-Sinc Downsample 2X packetized2', 38, [2, 4], kernelConservative)
        generateFilterPack('Gauss-Sinc Upsample 2X', 15, [2, 1], kernelBalanced)

        #
        # 1.5X
        #

        generateFilterPack('Gauss-Sinc Downsample 1.5X', 29, [2, 3], kernelConservative)
        generateFilterPack('Gauss-Sinc Upsample 1.5X', 16, [3, 2], kernelBalanced)
   
        #
        # 1.333X
        #

        generateFilterPack('Gauss-Sinc Downsample 1.333X', 26, [3, 4], kernelConservative)
        generateFilterPack('Gauss-Sinc Upsample 1.333X', 15, [4, 3], kernelBalanced)

        #
        # 1.25X
        #

        generateFilterPack('Gauss-Sinc Downsample 1.25X', 25, [4, 5], kernelConservative)
        generateFilterPack('Gauss-Sinc Upsample 1.25X', 16, [5, 4], kernelBalanced)

    #----------------------------------------------------------------
    #
    # Gauss kernel downsampling (for masks).
    #
    #----------------------------------------------------------------

    if False:

        targetSigma = 0.6

        gaussKernel = lambda sigma: (lambda x, normalized: gauss(x, sigma, normalized))

        initialSigma = targetSigma
        sustainingSigma = lambda factor: np.sqrt(targetSigma ** 2 - (targetSigma * factor) ** 2); # sigma for unscaled kernel

        #
        # 1.5X
        #

        generateFilterPack('Gauss-Mask Downsample Initial', 7, [2, 3], gaussKernel(initialSigma))
        generateFilterPack('Gauss-Mask Downsample Sustaining', 7, [2, 3], gaussKernel(sustainingSigma(2.0 / 3)))

        #
        # 1.33333X
        #

        generateFilterPack('Gauss-Mask Downsample Initial', 8, [3, 4], gaussKernel(initialSigma))
        generateFilterPack('Gauss-Mask Downsample Sustaining', 6, [3, 4], gaussKernel(sustainingSigma(3.0 / 4)))

        #
        # 1.25X
        #

        generateFilterPack('Gauss-Mask Downsample Initial', 9, [4, 5], gaussKernel(initialSigma))
        generateFilterPack('Gauss-Mask Downsample Sustaining', 7, [4, 5], gaussKernel(sustainingSigma(4.0 / 5)))

    #----------------------------------------------------------------
    #
    # Gradient by Gauss
    #
    #----------------------------------------------------------------

    if True:
    
        targetSigma = 0.6

        gradKernelDog = lambda x, normalized: -dog(x, targetSigma, normalized)
        gradKernelDiff = lambda x, normalized: -gaussDiff(x, targetSigma, normalized)

        gaussKernel = lambda x, normalized: gauss(x, targetSigma, normalized)

        generateFilterPack('Grad-Gauss Dog', 5, [1, 1], gradKernelDog)
        generateFilterPack('Grad-Gauss Diff', 5, [1, 1], gradKernelDiff)

        generateFilterPack('Gauss Across', 5, [1, 1], gaussKernel)
