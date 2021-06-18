#include "colorConversions.h"

#include "convertYuv420/convertYuvRgbFunc.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTool.h"

//================================================================
//
// convertColorPixelToMonoPixel
//
//================================================================

GPUTOOL_2D_BEG
(
    convertColorPixelToMonoPixel,
    PREP_EMPTY,
    ((const uint8_x4, src))
    ((uint8, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    float32 Y, Pb, Pr;
    convertBgrToYPbPr(loadNorm(src), Y, Pb, Pr);
    storeNorm(dst, 0.5f * Y + 0.5f); // from [-1, +1] to [0, 1].
}
#endif    
GPUTOOL_2D_END
