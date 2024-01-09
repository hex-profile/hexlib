#if HOSTCODE
#include "randomImage.h"
#endif

#include "gpuSupport/gpuTool.h"
#include "rndgen/rndgenBase.h"

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
