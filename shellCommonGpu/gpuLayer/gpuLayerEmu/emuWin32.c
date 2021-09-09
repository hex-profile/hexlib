#if defined(_MSC_VER)

#include "emuWin32.h"

#include "errorLog/debugBreak.h"
#include "storage/rememberCleanup.h"
#include "dataAlloc/arrayObjectMemory.inl"
#include "point3d/point3d.h"

namespace emuWin32 {

//================================================================
//
// EmuWin32ThreadConverter
//
//================================================================

EmuWin32ThreadConverter::EmuWin32ThreadConverter()
{
    fiberConvertThreadToFiber(fiber);
}

EmuWin32ThreadConverter::~EmuWin32ThreadConverter()
{
    if (fiberIsCreated(fiber))
        DEBUG_BREAK_CHECK(fiberConvertFiberToThread());
}

//================================================================
//
// FiberTask
//
//================================================================

struct FiberTask
{
    EmuKernelFunc* kernelFunc;
    const void* kernelParams;
    Fiber mainFiber;
    EmuWin32* emuSupport;

    EmuParams emuParams;
};

//================================================================
//
// fiberFunc
//
//================================================================

static void FIBER_CONVENTION fiberFunc(void* parameter)
{

    //
    // server loop
    //

    for (;;)
    {
        FiberTask& task = * (FiberTask*) parameter;
        EmuParams& emuParams = task.emuParams;
        EmuWin32& that = *task.emuSupport;

        Space fiberIdx = emuParams.fiberIdx;
        ARRAY_EXPOSE_EX(that.fibers, fibers);

        if (!that.fiberException) // may be already set by preceding fibers in this call
        {
            try
            {
                task.kernelFunc(task.kernelParams, const_cast<EmuParams&>(emuParams));
            }
            catch (const EmuError& location)
            {
                if (that.fiberException == 0)
                    that.fiberException = location ? location : CT("Emu: Fiber error");
            }
            catch (...)
            {
                that.fiberException = CT("Emu: Fiber error");
            }
        }

        //
        // error cleanup
        //

        ++that.fiberExitCount;

        if (that.fiberExitCount == that.fiberCount) // is it the last fiber?
            fiberSwitch(fibersPtr[fiberIdx], task.mainFiber);
        else
            that.switchToNextFiber(fiberIdx);

    }

}

//================================================================
//
// FiberOwner::create
// FiberOwner::destroy
//
//================================================================

bool FiberOwner::create(FiberFunc* func, void* param)
{
    destroy();

    // Windows doesn't want to allocate less than 64 kilobytes of stack.
    return fiberCreate(func, param, 64*1024, fiber);
}

//----------------------------------------------------------------

void FiberOwner::destroy()
{
    fiberDestroy(fiber);
}

//================================================================
//
// EmuWin32::create
//
//================================================================

stdbool EmuWin32::create(stdPars(CreateKit))
{
    //
    // Deallocate everything and set to zero
    //

    destroy();

    //
    // Create fiber tasks memory. Remember to deallocate array in case of error.
    //

    require(fiberTasks.realloc(EMU_MAX_THREAD_COUNT, cpuBaseByteAlignment, kit.malloc, stdPass));
    REMEMBER_CLEANUP_EX(fiberTasksCleanup, fiberTasks.dealloc());
    ARRAY_EXPOSE(fiberTasks);

    //
    // Create fibers. Remember to deallocate fibers in case of error.
    //

    require(fibers.realloc(EMU_MAX_THREAD_COUNT, kit.malloc, true, stdPass));
    REMEMBER_CLEANUP_EX(fibersCleanup, fibers.dealloc());
    ARRAY_EXPOSE(fibers);

    for_count (i, fibersSize) // can be interrupted in the middle by error
        REQUIRE(fibersPtr[i].create(fiberFunc, &fiberTasksPtr[i]));

    //
    // Create shared memory holder.
    // Both memory address and size should be aligned to maxNaturalAlignment.
    //

    COMPILE_ASSERT(EMU_MAX_SRAM_SIZE % maxNaturalAlignment == 0);
    require(sramHolder.realloc(EMU_MAX_SRAM_SIZE, maxNaturalAlignment, kit.malloc, stdPass));
    REMEMBER_CLEANUP_EX(sramCleanup, sramHolder.dealloc());

    //
    // Record success.
    //

    fiberTasksCleanup.cancel();
    fibersCleanup.cancel();
    sramCleanup.cancel();

    created = true;

    ////

    returnTrue;
}

//================================================================
//
// EmuWin32::destroy()
//
//================================================================

void EmuWin32::destroy()
{
    created = false;

    sramHolder.dealloc();
    fiberTasks.dealloc();
    fibers.dealloc();

    resetState();
}

//================================================================
//
// CHECK_RETURN
//
//================================================================

#define CHECK_RETURN_EX(condition, errorValue) \
    if (condition) ; else return (errorValue)

#define CHECK_RETURN(condition) \
    CHECK_RETURN_EX(condition, EMU_ERRMSG(condition))

//----------------------------------------------------------------

#define CHECK_RETURN_DBG_EX(condition, errorValue) \
    if (condition) ; else {DEBUG_BREAK_INLINE(); return (errorValue);}

#define CHECK_RETURN_DBG(condition) \
    CHECK_RETURN_DBG_EX(condition, EMU_ERRMSG(condition))

//----------------------------------------------------------------

#define CHECK_THROW_EX(condition, errorValue) \
    if (condition) ; else throw (errorValue)

//================================================================
//
// EmuWin32::launchKernel
//
//================================================================

EmuError EmuWin32::launchKernel
(
    EmuWin32ThreadConverter& threadConverter,
    GroupSpace groupSegmentOrg,
    GroupSpace groupSegmentSize,
    const Point3D<GroupSpace>& groupCount,
    const Point<Space>& threadCount,
    EmuKernelFunc* kernelFunc,
    const void* userParams
)
{
    CHECK_RETURN(created);

    CHECK_RETURN(threadCount.X >= 1 && threadCount.X <= EMU_MAX_THREAD_COUNT_X);
    CHECK_RETURN(threadCount.Y >= 1 && threadCount.Y <= EMU_MAX_THREAD_COUNT_Y);
    Space threadCountArea = threadCount.X * threadCount.Y;
    CHECK_RETURN(threadCountArea <= EMU_MAX_THREAD_COUNT);

    ////

    ARRAY_EXPOSE_UNSAFE(sramHolder);
    CHECK_RETURN_DBG(sramHolderSize == EMU_MAX_SRAM_SIZE);

    //
    // Setup fiber parameters
    //

    EmuSharedParams sharedParams
    (
        threadCount,
        point3D(0), // groupIdx
        convertExact<Space>(groupCount),
        *this
    );

    ////

    ARRAY_EXPOSE(fiberTasks);
    CHECK_RETURN_DBG(threadCountArea <= fiberTasksSize);

    CHECK_RETURN(threadConverter.created());
    auto& mainFiber = threadConverter.fiber;

    for_count (Y, threadCount.Y)
    {
        for_count (X, threadCount.X)
        {
            Space i = X + Y * threadCount.X;

            FiberTask& task = fiberTasksPtr[i];
            task.kernelFunc = kernelFunc;
            task.kernelParams = userParams;
            task.mainFiber = mainFiber;
            task.emuSupport = this;

            task.emuParams.sharedParams = &sharedParams;
            task.emuParams.fiberIdx = i;
            task.emuParams.threadIdx = point(X, Y);
        }
    }

    //
    //
    //

    ARRAY_EXPOSE(fibers);

    GroupSpace groupYZ = groupSegmentOrg / groupCount.X;
    GroupSpace groupX = groupSegmentOrg - groupYZ * groupCount.X;

    GroupSpace groupZ = groupYZ / groupCount.Y;
    GroupSpace groupY = groupYZ - groupZ * groupCount.Y;

    ////

    for_count (k, groupSegmentSize)
    {
        sharedParams.groupIdx = point3D(Space(groupX), Space(groupY), Space(groupZ));

        ////

        ++groupX;

        if (groupX == groupCount.X)
            {groupX = 0; ++groupY;}

        if (groupY == groupCount.Y)
            {groupY = 0; ++groupZ;}

        ////

        for_count (i, threadCountArea)
        {
            FiberTask& task = fiberTasksPtr[i];
            task.emuParams.sramAllocator.setup(CpuAddrU(sramHolderPtr), sramHolderSize, 1);
        }

        resetState(threadCountArea);
        fiberSwitch(mainFiber, fibersPtr[0]);

        CHECK_RETURN_DBG(fiberExitCount == fiberCount);
        CHECK_RETURN_EX(fiberException == 0, fiberException);
    }

    ////

    return 0;
}

//================================================================
//
// EmuWin32::switchToNextFiber
//
//================================================================

inline void EmuWin32::switchToNextFiber(Space fiberIdx)
{
    auto nextIdx = fiberIdx + 1;

    if (nextIdx == fiberCount)
        nextIdx = 0;

    ARRAY_EXPOSE(fibers);

    fiberSwitch(fibersPtr[fiberIdx], fibersPtr[nextIdx]);
}

//================================================================
//
// EmuWin32::syncThreads
//
//================================================================

void EmuWin32::syncThreads(Space fiberIdx, uint32 id, EmuError errMsg)
{

    //
    // Emergency exit, if some fiber error-exited.
    //

    CHECK_THROW_EX(fiberException == 0, fiberException);

    //
    // Fiber 0 starts syncing, other fibers go in sequence.
    //

    if (fiberIdx == 0)
    {
        CHECK_THROW_EX(syncthreadsId == 0, errMsg);
        syncthreadsId = id;
    }
    else
    {
        CHECK_THROW_EX(id == syncthreadsId, errMsg);
    }

    //
    // Let the next fibers reach this sync point
    //

    switchToNextFiber(fiberIdx);

    //
    // Emergency exit, if some fiber error-exited.
    //

    CHECK_THROW_EX(fiberException == 0, fiberException);

    //
    // Fiber 0 is first AFTER the sync point
    // All fibers should take part im the sync!
    //

    if (fiberIdx == 0)
    {
        CHECK_THROW_EX(fiberExitCount == 0, errMsg);
        syncthreadsId = 0;
    }

}

//================================================================
//
// EmuWin32::fatalError
//
//================================================================

void EmuWin32::fatalError(EmuError errMsg)
{
    DEBUG_BREAK_INLINE();
    throw errMsg;
}

//----------------------------------------------------------------

}

#endif
