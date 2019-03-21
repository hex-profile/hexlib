#include "yuvFile.h"

#include "cpuFuncKit.h"
#include "errorLog/errorLog.h"
#include "formattedOutput/requireMsg.h"
#include "userOutput/printMsg.h"

namespace yuvFile {

//================================================================
//
// YuvFile::setupForInput
//
//================================================================

template <typename RawPixel>
bool YuvFile<RawPixel>::setup(BinaryInputStream* inputStream, BinaryOutputStream* outputStream, FilePositioning* filePositioning, const Point<Space>& frameSize, stdPars(DiagnosticKit))
{
    stdBegin;

    ////

    REQUIRE_MSG(yuv420SizeValid(frameSize), STR("YUV video frame size is not valid"));
    Space frameBytes = sizeof(RawPixel) * yuv420TotalArea(frameSize);
    REQUIRE(frameBytes >= 1);
    Space frameBytesU = frameBytes;

    ////

    int32 frameCount = 0;
    int32 frameIndex = 0;

    if (filePositioning)
    {
        uint64 fileSize = filePositioning->getSize();

        uint64 frameCount64 = fileSize / frameBytesU;
        REQUIRE_MSG(frameCount64 * frameBytesU == fileSize, STR("YUV video file size is not a multiple of the frame size"));

        REQUIRE_MSG(frameCount64 <= 0x7FFFFFFF, STR("YUV video file is too big"));
        frameCount = int32(frameCount64);

        ////

        uint64 filePos = filePositioning->getPosition();
        REQUIRE(filePos <= fileSize);

        uint64 framePos64 = filePos / frameBytesU;
        REQUIRE_MSG(framePos64 * frameBytesU == filePos, STR("YUV video file position is not a multiple of the frame size"));
        frameIndex = int32(framePos64);
    }

    ////

    theInputStream = inputStream;
    theOutputStream = outputStream;
    theFilePositioning = filePositioning;

    theFrameSize = frameSize;
    theFrameBytes = frameBytes;

    theFrameCount = frameCount;
    theFrameIndex = frameIndex;

    stdEnd;
}

//================================================================
//
// YuvFile::setPosition
//
//================================================================

template <typename RawPixel>
bool YuvFile<RawPixel>::setPosition(int32 frameIndex, stdPars(DiagnosticKit))
{
    stdBegin;

    REQUIRE(frameIndex >= 0 && frameIndex <= theFrameCount);

    REQUIRE(theFilePositioning);

    if (frameIndex != theFrameIndex)
        require(theFilePositioning->setPosition(uint64(frameIndex) * uint64(theFrameBytes), stdPass));

    theFrameIndex = frameIndex;

    stdEnd;
}

//================================================================
//
// YuvFile::readFrame
//
//================================================================

template <typename RawPixel>
bool YuvFile<RawPixel>::readFrame(const Array<RawPixel>& frame, stdPars(DiagnosticKit))
{
    stdBegin;

    REQUIRE(frame.size() == yuv420TotalArea(theFrameSize));

    ////

    ARRAY_EXPOSE_UNSAFE(frame, frame);

    REQUIRE(theInputStream);
    require(theInputStream->read(framePtr, sizeof(RawPixel) * frameSize, stdPass));
    ++theFrameIndex;

    stdEnd;
}

//================================================================
//
// YuvFile::writeFrame
//
//================================================================

template <typename RawPixel>
bool YuvFile<RawPixel>::writeFrame(const Array<const RawPixel>& frame, stdPars(DiagnosticKit))
{
    stdBegin;

    REQUIRE(frame.size() == yuv420TotalArea(theFrameSize));

    ////

    ARRAY_EXPOSE_UNSAFE(frame, frame);

    REQUIRE(theOutputStream);
    require(theOutputStream->write(framePtr, sizeof(RawPixel) * frameSize, stdPass));
    ++theFrameIndex;

    theFrameCount = maxv(theFrameCount, theFrameIndex);

    stdEnd;
}

//----------------------------------------------------------------

template class YuvFile<uint8>;
template class YuvFile<uint16>;

//----------------------------------------------------------------

}
