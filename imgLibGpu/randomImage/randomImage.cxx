#if HOSTCODE
#include "randomImage.h"
#endif

#include "gpuSupport/gpuTool.h"
#include "rndgen/rndgenBase.h"

//================================================================
//
// initializeRandomStateArray
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    initializeRandomStateArray,
    (256, 1), false,
    PREP_EMPTY,
    ((RndgenState, state)),
    ((uint32, frameIndex))
    ((uint32, xorValue))
)
#if DEVCODE
{
    uint32 finalIndex = X + Y * vGlobSize.X + frameIndex * (vGlobSize.X * vGlobSize.Y);
    *state = distributiveHash(finalIndex ^ xorValue);
}
#endif
GPUTOOL_2D_END_EX

//================================================================
//
// initializeRandomStateMatrix
//
//================================================================

GPUTOOL_2D_BEG
(
    initializeRandomStateMatrix,
    PREP_EMPTY,
    ((RndgenState, state)),
    ((uint32, frameIndex))
    ((uint32, xorValue))
)
#if DEVCODE
{
    uint32 finalIndex = X + Y * vGlobSize.X + frameIndex * (vGlobSize.X * vGlobSize.Y);
    *state = distributiveHash(finalIndex ^ xorValue);
}
#endif
GPUTOOL_2D_END
