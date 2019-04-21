#include "frameChange.h"

#include <memory.h>

#include "errorLog/errorLog.h"

namespace frameChange {

//================================================================
//
// resetFrameDesc
//
//================================================================

void resetFrameDesc(FrameDesc& desc)
{
    desc.initialized = false;
    desc.name.resizeNull();
    desc.frameIndex = 0;
    desc.frameCount = 0;
    desc.interlacedMode = 0;
    desc.interlacedLower = 0;
    desc.imageSize = point(0);
}

//================================================================
//
// updateFrameState
//
//================================================================

stdbool updateFrameState(const AtVideoInfo& info, FrameDesc& desc, stdPars(FrameChangeDetector::Kit))
{
    stdBegin;

    ////

    Space nameSize = 0;
    REQUIRE(convertExact(info.videofileName.size, nameSize));

    if_not (desc.name.resize(nameSize))
        require(desc.name.realloc(nameSize, cpuBaseByteAlignment, kit.malloc, stdPass));

    ARRAY_EXPOSE_UNSAFE(desc.name, descName);
    memcpy(descNamePtr, info.videofileName.ptr, nameSize * sizeof(CharType));

    ////

    desc.frameIndex = info.frameIndex;
    desc.frameCount = info.frameCount;

    ////

    desc.interlacedMode = info.interlacedMode;
    desc.interlacedLower = info.interlacedLower;

    ////

    desc.imageSize = info.frameSize;

    ////

    desc.initialized = true;

    stdEnd;
}

//================================================================
//
// frameNameEqual
//
//================================================================

bool frameNameEqual(const Array<const CharType>& a, const Array<const CharType>& b)
{
    ARRAY_EXPOSE(a);
    ARRAY_EXPOSE(b);

    ////

    require(aSize == bSize);
    Space size = aSize;

    ////

    uint32 orMask = 0;

    for (Space i = 0; i < size; ++i)
        orMask |= (aPtr[i] ^ bPtr[i]);

    require(orMask == 0);

    return true;
}

//================================================================
//
// frameStateEqual
//
//================================================================

bool frameStateEqual(const FrameDesc& a, const FrameDesc& b)
{
    require(frameNameEqual(a.name, b.name));

    require(a.frameIndex == b.frameIndex);
    require(a.frameCount == b.frameCount);

    require(a.interlacedMode == b.interlacedMode);
    require(a.interlacedLower == b.interlacedLower);

    require(a.imageSize == b.imageSize);

    return true;
}

//================================================================
//
// FrameChangeDetector::reset
//
//================================================================

void FrameChangeDetector::reset()
{
    resetFrameDesc(frameHist[0]);
    resetFrameDesc(frameHist[1]);
    histIdx = 0;
}

//================================================================
//
// FrameChangeDetector::check
//
//================================================================

stdbool FrameChangeDetector::check(const AtVideoInfo& info, bool& frameAdvance, stdPars(Kit))
{
    stdBegin;

    FrameDesc& prevFrame = frameHist[histIdx ^ 0];
    FrameDesc& currFrame = frameHist[histIdx ^ 1];

    ////

    require(updateFrameState(info, currFrame, stdPass));

    ////

    bool sameFrame =
        currFrame.initialized && prevFrame.initialized
        && frameStateEqual(currFrame, prevFrame);

    frameAdvance = !sameFrame;

    ////

    histIdx ^= 1;

    stdEnd;
}

//----------------------------------------------------------------

}
