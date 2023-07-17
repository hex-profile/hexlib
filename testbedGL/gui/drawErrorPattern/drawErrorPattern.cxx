#if HOSTCODE
#include "drawErrorPattern.h"
#endif

#include "gpuSupport/gpuTool.h"

//================================================================
//
// pattern*
//
//================================================================

static constexpr int patternSizeX = 53;
static constexpr int patternSizeY = 8;

////

#if DEVCODE

devConstant char patternData[] =
{
    "OO   OO    O   O   O     OOO  OO   OO    O   OO      "
    "O O  O O  O O  O   O     O    O O  O O  O O  O O     "
    "O O  OO   O O  O O O     OO   OO   OO   O O  OO      "
    "O O  OO   OOO  O O O     O    OO   OO   O O  OO      "
    "OO   O O  O O   OOO      OOO  O O  O O   O   O O     "
    "                                                     "
    "                                                     "
    "                                                     "
};

COMPILE_ASSERT(sizeof(patternData) == patternSizeX * patternSizeY + 1);

#endif

//================================================================
//
// drawErrorPattern
//
//================================================================

GPUTOOL_2D_BEG
(
    drawErrorPattern,
    PREP_EMPTY,
    ((uint8_x4, dstImage)),
    PREP_EMPTY
)
#if DEVCODE
{
    auto pX = SpaceU(X) % SpaceU(patternSizeX);
    auto pY = SpaceU(Y) % SpaceU(patternSizeY);

    auto color = make_uint8_x4(0xFF, 0xFF, 0xFF, 0);

    if (patternData[pX + pY * patternSizeX] == 'O')
        color = make_uint8_x4(0x00, 0x00, 0xFF, 0);

    *dstImage = color;
}
#endif
GPUTOOL_2D_END
