#pragma once

#include "dataAlloc/arrayMemory.h"
#include "point/point.h"
#include "cpuFuncKit.h"
#include "atInterface/atInterface.h"
#include "allocation/mallocKit.h"

namespace frameChange {

//================================================================
//
// FrameDesc
//
//================================================================

struct FrameDesc
{
    bool initialized = false;

    // Videofile name
    ArrayMemory<CharType> name;

    // Position in videofile
    Space frameIndex = 0;
    Space frameCount = 0;

    // Interlace params
    int32 interlacedMode = 0;
    int32 interlacedLower = 0;

    // Video image size
    Point<Space> imageSize;
};

//================================================================
//
// FrameChangeDetector
//
//================================================================

class FrameChangeDetector
{

public:

    using Kit = KitCombine<ErrorLogKit, MallocKit>;

    void reset();
    void check(const AtVideoInfo& info, bool& frameAdvance, stdPars(Kit));

private:

    int32 histIdx = 0;
    FrameDesc frameHist[2];

};

//----------------------------------------------------------------

}

using frameChange::FrameChangeDetector;
