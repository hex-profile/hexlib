#pragma once

#include "dataAlloc/arrayMemory.h"
#include "cpuFuncKit.h"
#include "binaryFile/binaryFile.h"
#include "point/point.h"
#include "yuv420/yuv420Tools.h"
#include "numbers/interface/exchangeInterface.h"
#include "userOutput/diagnosticKit.h"

namespace yuvFile {

//================================================================
//
// YuvFile
//
//================================================================

template <typename RawPixel>
class YuvFile
{

public:

    // Any of the interfaces may be omitted
    stdbool setup(BinaryInputStream* inputStream, BinaryOutputStream* outputStream, FilePositioning* filePositioning, const Point<Space>& frameSize, stdPars(DiagnosticKit));

    void reset()
    {
        theInputStream = 0;
        theOutputStream = 0;
        theFilePositioning = 0;

        theFrameSize = point(0);
        theFrameBytes = 0;

        theFrameCount = 0;
        theFrameIndex = 0;
    }

public:

    Point<Space> frameSize() const {return theFrameSize;}
    Space frameBytes() const {return theFrameBytes;}

public:

    int32 frameCount() const {return theFrameCount;}
    int32 frameIndex() const {return theFrameIndex;}

public:

    stdbool setPosition(int32 frameIndex, stdPars(DiagnosticKit));

public:

    stdbool readFrame(const Array<RawPixel>& frame, stdPars(DiagnosticKit));
    stdbool writeFrame(const Array<const RawPixel>& frame, stdPars(DiagnosticKit));

public:

    friend inline void exchange(YuvFile& A, YuvFile& B)
    {
        ::exchange(A.theInputStream, B.theInputStream);
        ::exchange(A.theOutputStream, B.theOutputStream);
        ::exchange(A.theFilePositioning, B.theFilePositioning);

        ::exchange(A.theFrameSize, B.theFrameSize);
        ::exchange(A.theFrameBytes, B.theFrameBytes);

        ::exchange(A.theFrameCount, B.theFrameCount);
        ::exchange(A.theFrameIndex, B.theFrameIndex);
    }

private:

    BinaryInputStream* theInputStream = 0;
    BinaryOutputStream* theOutputStream = 0;
    FilePositioning* theFilePositioning = 0;

    Point<Space> theFrameSize = point(0);
    Space theFrameBytes = 0;

    int32 theFrameCount = 0;
    int32 theFrameIndex = 0;

};

//----------------------------------------------------------------

}

using yuvFile::YuvFile;
