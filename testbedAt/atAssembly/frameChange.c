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

void updateFrameState(const AtVideoInfo& info, FrameDesc& desc, stdPars(FrameChangeDetector::Kit))
{
    Space nameSize = 0;
    REQUIRE(convertExact(info.videofileName.size, nameSize));

    if_not (desc.name.resize(nameSize))
        desc.name.realloc(nameSize, cpuBaseByteAlignment, kit.malloc, stdPass);

    ARRAY_EXPOSE_UNSAFE_EX(desc.name, descName);
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

    ensure(aSize == bSize);
    Space size = aSize;

    ////

    uint32 orMask = 0;

    for_count (i, size)
        orMask |= (aPtr[i] ^ bPtr[i]);

    ensure(orMask == 0);

    return true;
}

//================================================================
//
// frameStateEqual
//
//================================================================

bool frameStateEqual(const FrameDesc& a, const FrameDesc& b)
{
    ensure(frameNameEqual(a.name, b.name));

    ensure(a.frameIndex == b.frameIndex);
    ensure(a.frameCount == b.frameCount);

    ensure(a.interlacedMode == b.interlacedMode);
    ensure(a.interlacedLower == b.interlacedLower);

    ensure(a.imageSize == b.imageSize);

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

void FrameChangeDetector::check(const AtVideoInfo& info, bool& frameAdvance, stdPars(Kit))
{
    FrameDesc& prevFrame = frameHist[histIdx ^ 0];
    FrameDesc& currFrame = frameHist[histIdx ^ 1];

    ////

    updateFrameState(info, currFrame, stdPass);

    ////

    bool sameFrame =
        currFrame.initialized && prevFrame.initialized
        && frameStateEqual(currFrame, prevFrame);

    frameAdvance = !sameFrame;

    ////

    histIdx ^= 1;
}

//----------------------------------------------------------------

}
