#pragma once

#include "data/matrix.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/diagnosticKit.h"
#include "vectorTypes/vectorBase.h"

namespace bmpFile {

//================================================================
//
// Kit
//
//================================================================

using Kit = DiagnosticKit;

//================================================================
//
// BmpWriter
//
//================================================================

class BmpWriter
{

public:

    ~BmpWriter() {clearMemory();}

public:

    void write(const MatrixAP<const uint8>& image, const CharArray& filename, bool disableSlowdown, stdPars(Kit));
    void write(const MatrixAP<const uint8_x4>& image, const CharArray& filename, bool disableSlowdown, stdPars(Kit));

public:

    void clearMemory();

private:

    template <typename Pixel>
    void writeFunc(const MatrixAP<const Pixel>& image, const CharArray& filename, bool disableSlowdown, stdPars(Kit));

private:

    void* memoryPtr = 0;
    size_t memorySize = 0;

};

//----------------------------------------------------------------

}
