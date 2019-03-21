#pragma once

#include "gpuLayer/gpuLayerEmu/emuWin32.h"
#include "dataAlloc/arrayObjMem.h"
#include "kit/kit.h"
#include "allocation/mallocKit.h"
#include "point3d/point3dBase.h"
#include "interfaces/threadManagerKit.h"

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

KIT_COMBINE3(CreateKit, ErrorLogKit, MallocKit, ThreadManagerKit);

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

    bool create(Space streamCount, stdPars(CreateKit));
    void destroy();
    bool created() const {return serverArray.allocated();}

    Space threadCount() const {return serverArray.size();}

public:

    bool launchKernel
    (
        const Point3D<Space>& groupCount,
        const Point<Space>& threadCount,
        EmuKernelFunc* kernel,
        const void* userParams,
        stdPars(ErrorLogKit)
    );

private:

    ArrayObjMem<class ServerKeeper> serverArray;

};

//----------------------------------------------------------------

}
