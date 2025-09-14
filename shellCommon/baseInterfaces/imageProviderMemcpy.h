#pragma once

#include "baseInterfaces/baseImageConsole.h"
#include "errorLog/errorLogKit.h"
#include "dataAlloc/cpuDefaultAlignments.h"
#include "errorLog/errorLog.h"

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

    ImageProviderMemcpy(const MatrixAP<const Pixel>& source, const ErrorLogKit& kit)
        : source(source), kit(kit) {}

public:

    Space desiredPitch() const
        {return source.memPitch();}

    Space desiredBaseByteAlignment() const
        {return cpuBaseByteAlignment;}

    stdbool saveBgr32(const MatrixAP<Pixel>& dest, stdNullPars);

    stdbool saveBgr24(const MatrixAP<uint8>& dest, stdNullPars)
        {REQUIRE(false); returnTrue;}

private:

    MatrixAP<const Pixel> source;
    ErrorLogKit kit;

};
