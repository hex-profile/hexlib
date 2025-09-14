#pragma once

#include "gpuLayer/gpuLayerEmu/emuWin32.h"
#include "dataAlloc/arrayObjectMemory.h"
#include "kit/kit.h"
#include "allocation/mallocKit.h"
#include "point3d/point3dBase.h"

namespace emuMultiWin32 {

using namespace emuWin32;

//================================================================
//
// getCpuCount
//
// Returns the number of CPU cores present in the system.
//
//================================================================

Space getCpuCount();

//================================================================
//
// CreateKit
//
//================================================================

using CreateKit = KitCombine<ErrorLogKit, MallocKit>;

//================================================================
//
// EmuMultiWin32
//
//================================================================

class EmuMultiWin32
{

public:

    EmuMultiWin32();
    ~EmuMultiWin32();

public:

    void create(Space streamCount, stdPars(CreateKit));
    void destroy();
    bool created() const {return serverArray.allocated();}

    Space threadCount() const {return serverArray.size();}

public:

    void launchKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        EmuKernelFunc* kernel,
        const void* userParams,
        stdPars(ErrorLogKit)
    );

private:

    ArrayObjectMemory<class ServerKeeper> serverArray;

};

//----------------------------------------------------------------

}
