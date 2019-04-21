#pragma once

#include "point3d/point3dBase.h"
#include "errorLog/errorLogKit.h"
#include "data/space.h"
#include "stdFunc/stdFunc.h"
#include "point/point.h"
#include "gpuDevice/gpuDeviceEmu.h"

namespace emuMultiProcFake {

//================================================================
//
// getCpuCount
//
//================================================================

inline Space getCpuCount()
{
    return 1;
}

//================================================================
//
// properties
//
//================================================================

const Space EMU_MAX_THREAD_COUNT_X = 512;
const Space EMU_MAX_THREAD_COUNT_Y = 512;
const Space EMU_MAX_THREAD_COUNT = 1024;

const Space EMU_MAX_SRAM_SIZE = 65536;

//================================================================
//
// CreateKit
//
//================================================================

KIT_COMBINE1(CreateKit, ErrorLogKit);

//================================================================
//
// EmuMultiProcFake
//
//================================================================

class EmuMultiProcFake
{

public:

    stdbool create(Space streamCount, stdPars(CreateKit))
        {return true;}

    void destroy()
        {}

    bool created() const
        {return true;}

    stdbool launchKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        EmuKernelFunc* kernel,
        const void* userParams,
        stdPars(ErrorLogKit)
    )
    {
        REQUIRE(false); // not implemented
        return false;
    }

};

//----------------------------------------------------------------

}
