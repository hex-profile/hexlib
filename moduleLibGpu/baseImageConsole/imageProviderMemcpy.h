#pragma once

#include "baseImageConsole/baseImageConsole.h"
#include "errorLog/errorLogKit.h"
#include "dataAlloc/cpuDefaultAlignments.h"

//================================================================
//
// ImageProviderMemcpy
//
//================================================================

class ImageProviderMemcpy : public BaseImageProvider
{

public:

    using Pixel = uint8_x4;

public:

    ImageProviderMemcpy(const Matrix<const Pixel>& source, const ErrorLogKit& kit)
        : source(source), kit(kit) {}

public:

    Space getPitch() const
        {return source.memPitch();}

    Space baseByteAlignment() const
        {return cpuBaseByteAlignment;}

    stdbool saveImage(const Matrix<Pixel>& dest, stdNullPars);

private:

    Matrix<const Pixel> source;
    ErrorLogKit kit;

};
