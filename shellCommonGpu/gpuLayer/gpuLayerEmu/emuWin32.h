#pragma once

#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"
#include "gpuDevice/gpuDeviceEmu.h"
#include "dataAlloc/arrayMemory.h"
#include "dataAlloc/arrayObjectMemory.h"
#include "allocation/mallocKit.h"
#include "fibers/fibers.h"

namespace emuWin32 {

using namespace fibers;

//================================================================
//
// emuProps
//
//================================================================

const Space EMU_MAX_THREAD_COUNT_X = 512;
const Space EMU_MAX_THREAD_COUNT_Y = 512;
const Space EMU_MAX_THREAD_COUNT = 1024;

const Space EMU_MAX_SRAM_SIZE = 65536;

//================================================================
//
// EmuWin32ThreadConverter
//
//================================================================

class EmuWin32ThreadConverter
{

    friend class EmuWin32;

public:

    EmuWin32ThreadConverter();
    ~EmuWin32ThreadConverter();
    bool created() const {return fiberIsCreated(fiber);}

private:

    Fiber fiber;

};

//================================================================
//
// GroupSpace
//
// Should be able to hold the area of the biggest group count.
//
//================================================================

using GroupSpace = uint32;

//================================================================
//
// FiberOwner
//
//================================================================

class FiberOwner
{

public:

    inline ~FiberOwner() {destroy();}

public:

    bool create(FiberFunc* func, void* param);
    void destroy();

    operator Fiber& () {return fiber;}

private:

    Fiber fiber;

};

//================================================================
//
// EmuWin32
//
//================================================================

class EmuWin32 : public EmuKernelTools
{

public:

    using CreateKit = KitCombine<ErrorLogKit, MallocKit>;

    void create(stdPars(CreateKit));
    void destroy();

    inline EmuWin32()
    {
        fiberCount = 0;
        syncthreadsId = 0;
        fiberExitCount = 0;
        fiberException = 0;
    }

    inline ~EmuWin32()
    {
        destroy();
    }

public:

    EmuError launchKernel
    (
        EmuWin32ThreadConverter& threadConverter,
        GroupSpace groupSegmentOrg, GroupSpace groupSegmentSize,
        const Point3D<GroupSpace>& groupCount,
        const Point<Space>& threadCount,
        EmuKernelFunc* kernelFunc,
        const void* userParams
    );

public:

    void syncThreads(Space fiberIdx, uint32 id, EmuError errMsg);
    void fatalError(EmuError errMsg);

private:

    inline void switchToNextFiber(Space fiberIdx);

private:

    friend void FIBER_CONVENTION fiberFunc(void* parameter);

private:

    bool created = false;

    ArrayObjectMemory<class FiberOwner> fibers;
    ArrayMemory<struct FiberTask> fiberTasks;
    ArrayMemory<Byte> sramHolder;
    ArrayMemory<EmuMaxWarpIntrinsicsType> warpIntrinsicsMemoryHolder;

    // current state
    Space fiberCount;
    uint32 syncthreadsId;
    Space fiberExitCount;
    EmuError fiberException;

    void resetState(Space threadCount = 0)
    {
        fiberCount = threadCount;
        syncthreadsId = 0;
        fiberExitCount = 0;
        fiberException = 0;
    }

};

//----------------------------------------------------------------

}

using emuWin32::EmuWin32;
using emuWin32::EmuWin32ThreadConverter;
