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

    void saveBgr32(const MatrixAP<Pixel>& dest, stdParsNull);

    void saveBgr24(const MatrixAP<uint8>& dest, stdParsNull)
        {REQUIRE(false);}

private:

    MatrixAP<const Pixel> source;
    ErrorLogKit kit;

};
