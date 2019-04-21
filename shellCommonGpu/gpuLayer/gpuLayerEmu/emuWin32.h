#pragma once

#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"
#include "gpuDevice/gpuDeviceEmu.h"
#include "dataAlloc/arrayMemory.h"
#include "dataAlloc/arrayObjMem.h"
#include "allocation/mallocKit.h"

namespace emuWin32 {

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
    bool created() {return fiber != 0;}

private:

    void* fiber;

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
// FiberFunc
//
//================================================================

typedef void __stdcall FiberFunc(void* param);

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

    operator void* () const {return fiber;}

private:

    void* fiber = 0;

};

//================================================================
//
// EmuWin32
//
//================================================================

class EmuWin32 : public EmuKernelTools
{

public:

    KIT_COMBINE2(CreateKit, ErrorLogKit, MallocKit);

    stdbool create(stdPars(CreateKit));
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

    friend void __stdcall fiberFunc(void* parameter);

private:

    bool created = false;

    ArrayObjMem<class FiberOwner> fibers;
    ArrayMemory<struct FiberTask> fiberTasks;
    ArrayMemory<Byte> sramHolder;

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
