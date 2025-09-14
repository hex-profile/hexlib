#if defined(_WIN32)

#include "emuMultiWin32.h"

#include <process.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "errorLog/debugBreak.h"
#include "dataAlloc/arrayObjectMemory.inl"
#include "storage/rememberCleanup.h"
#include "data/matrix.h"
#include "numbers/safeuint32/safeuint32.h"
#include "point3d/point3d.h"
#include "interfaces/syncObjects.h"
#include "threads/threads.h"
#include "gpuSupport/uniformPartition.h"

namespace emuMultiWin32 {

//================================================================
//
// getCpuCount
//
//================================================================

Space getCpuCount()
{
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return clampMin<uint32>(info.dwNumberOfProcessors, 1);
}

//================================================================
//
// ServerTask
//
// Thread server task structure.
// The data is changing during server loop.
//
//================================================================

struct ServerTask
{
    EmuError returnValueEx;
    GroupSpace groupSegmentOrg;
    GroupSpace groupSegmentSize;
    Point3D<GroupSpace> groupCount;
    Point<Space> threadCount;
    EmuKernelFunc* kernel; // == 0 on exit
    const void* userParams;

    inline void setup
    (
        EmuError returnValueEx,
        GroupSpace groupSegmentOrg,
        GroupSpace groupSegmentSize,
        Point3D<GroupSpace> groupCount,
        Point<Space> threadCount,
        EmuKernelFunc* kernel,
        const void* userParams
    )
    {
        this->returnValueEx = returnValueEx;
        this->groupSegmentOrg = groupSegmentOrg;
        this->groupSegmentSize = groupSegmentSize;
        this->groupCount.X = groupCount.X;
        this->groupCount.Y = groupCount.Y;
        this->groupCount.Z = groupCount.Z;
        this->threadCount.X = threadCount.X;
        this->threadCount.Y = threadCount.Y;
        this->kernel = kernel;
        this->userParams = userParams;
    }

    inline void setZero()
    {
        setup(0, 0, 0, point3D(GroupSpace(0)), point(0), 0, 0);
    }

    inline ServerTask()
    {
        setZero();
    }
};

//================================================================
//
// ServerTools
//
// Thread server read-only state structure.
// The data is not changing during server loop.
//
//================================================================

struct ServerTools
{
    EventOwner startEvent;
    EventOwner finishEvent;
    EmuWin32* coreEmulator = 0;
};

//================================================================
//
// ServerMemory
//
//================================================================

struct ServerMemory
{
    ServerTask task;
    ServerTools tools;
};

//================================================================
//
// serverFunc
//
//================================================================

static void serverFunc(void* params)
{
    EmuWin32ThreadConverter threadConverter;

    for (; ;)
    {
        ServerMemory& mem = * (ServerMemory*) params;

        //
        // wait for task
        //

        mem.tools.startEvent.wait();

        //
        // exit?
        //

        ServerTask& task = mem.task;
        if (task.kernel == 0) break;

        //
        // perform task
        //

        task.returnValueEx = mem.tools.coreEmulator->launchKernel
        (
            threadConverter,
            task.groupSegmentOrg,
            task.groupSegmentSize,
            point3D(GroupSpace(task.groupCount.X), GroupSpace(task.groupCount.Y), GroupSpace(task.groupCount.Z)),
            point(Space(task.threadCount.X), Space(task.threadCount.Y)),
            task.kernel,
            task.userParams
        );

        //
        // signal task completion
        //

        mem.tools.finishEvent.set();
    }
}

//================================================================
//
// ServerKeeper
//
//================================================================

class ServerKeeper : private ServerMemory
{

public:

    inline ServerKeeper()
        {}

    inline ~ServerKeeper()
        {destroy();}

    void create(stdPars(CreateKit));
    void destroy();
    bool created() {return threadControl.created();}

public:

    ServerTask& taskMemory()
        {return task;}

    inline void setStartEvent()
        {tools.startEvent.set();}

    inline void waitFinishEvent()
        {tools.finishEvent.wait();}

private:

    ThreadControl threadControl;
    EmuWin32 coreEmulator;

};

//================================================================
//
// ServerKeeper::create
//
//================================================================

void ServerKeeper::create(stdPars(CreateKit))
{
    //
    // Deallocate and set to zero
    //

    destroy();

    //
    // Create core emulator
    //

    coreEmulator.create(stdPass);
    REMEMBER_CLEANUP_EX(coreEmulatorCleanup, coreEmulator.destroy());

    //
    // Allocate events
    //

    eventCreate(tools.startEvent, stdPass);
    REMEMBER_CLEANUP_EX(startEventCleanup, tools.startEvent.clear());

    eventCreate(tools.finishEvent, stdPass);
    REMEMBER_CLEANUP_EX(finishEventCleanup, tools.finishEvent.clear());

    tools.coreEmulator = &coreEmulator;
    REMEMBER_CLEANUP_EX(emulatorPtrCleanup, tools.coreEmulator = 0);

    //
    // Launch thread (should be the last)
    //

    ServerMemory* serverMemory = this;
    threadCreate(serverFunc, serverMemory, 65536, threadControl, stdPass);

    //
    // Lower the thread priority.
    //

    REQUIRE(threadControl->setPriority(ThreadPriorityMinus1));

    //
    // Record success
    //

    coreEmulatorCleanup.cancel();
    startEventCleanup.cancel();
    finishEventCleanup.cancel();
    emulatorPtrCleanup.cancel();
}

//================================================================
//
// ServerKeeper::destroy
//
//================================================================

void ServerKeeper::destroy()
{

    //
    // If the thread is running, issue "exit" task,
    // wait for the thread shutdown and close the thread handle.
    //

    if (threadControl.created())
    {
        task.kernel = 0; // Issue "exit" task
        tools.startEvent.set();

        threadControl.waitAndClear();
    }

    //
    // Free core emulator
    //

    coreEmulator.destroy();

    //
    // Free events
    //

    tools.startEvent.clear();
    tools.finishEvent.clear();
    tools.coreEmulator = 0;

    //
    // Set everything to zero
    //

    task.setZero();

}

//================================================================
//
// EmuMultiWin32::EmuMultiWin32
// EmuMultiWin32::~EmuMultiWin32
//
//================================================================

EmuMultiWin32::EmuMultiWin32()
{
}

EmuMultiWin32::~EmuMultiWin32()
{
}

//================================================================
//
// EmuMultiWin32::destroy
//
//================================================================

void EmuMultiWin32::destroy()
{
    serverArray.dealloc();
}

//================================================================
//
// EmuMultiWin32::create
//
//================================================================

void EmuMultiWin32::create(Space streamCount, stdPars(CreateKit))
{
    destroy();

    //
    // Allocate memory. For the case of error, remember to destroy
    // the created servers (wait thread and destroy, by element destructors).
    //

    serverArray.reallocInHeap(streamCount, stdPass);
    REMEMBER_CLEANUP_EX(serverArrayCleanup, serverArray.dealloc());

    //
    // Create thread servers (can be interrupted by error).
    //

    ARRAY_EXPOSE(serverArray);

    for_count (i, streamCount)
        {serverArrayPtr[i].create(stdPass);}

    //
    // Record success.
    //

    serverArrayCleanup.cancel();
}

//================================================================
//
// EmuMultiWin32::launchKernel
//
//================================================================

void EmuMultiWin32::launchKernel
(
    const Point3D<Space>& groupCount,
    const Point<Space>& threadCount,
    EmuKernelFunc* kernel,
    const void* userParams,
    stdPars(ErrorLogKit)
)
{
    REQUIRE(created());

    //
    //
    // Check group count.
    //

    REQUIRE(groupCount >= 0);

    if_not (allv(groupCount >= 1))
        return;

    //
    // Give tasks to all thread servers, using uniform partition.
    //

    ARRAY_EXPOSE(serverArray);
    REQUIRE(serverArraySize >= 1);
    Space streamCount = serverArraySize;

    ////

    GroupSpace groupCountArea2D = 0;
    REQUIRE(safeMul(GroupSpace(groupCount.X), GroupSpace(groupCount.Y), groupCountArea2D));

    GroupSpace groupCountArea = 0;
    REQUIRE(safeMul(groupCountArea2D, GroupSpace(groupCount.Z), groupCountArea));

    ////

    UniformPartition<GroupSpace> partition(groupCountArea, GroupSpace(streamCount));

    ////

    for_count (k, GroupSpace(streamCount))
    {
        ServerKeeper& srv = serverArrayPtr[k];

        srv.taskMemory().setup
        (
            0,
            partition.nthOrg(k),
            partition.nthSize(k),
            convertExact<GroupSpace>(groupCount),
            threadCount,
            kernel,
            userParams
        );

        srv.setStartEvent();
    }

    //
    // Wait for all thread servers
    //

    EmuError recordedError = 0;

    for_count (k, streamCount)
    {
        ServerKeeper& srv = serverArrayPtr[k];
        srv.waitFinishEvent();

        EmuError srvError = srv.taskMemory().returnValueEx;

        if (srvError != 0 && recordedError == 0)
            recordedError = srvError;
    }

    ////

    REQUIRE_EX(recordedError == 0, kit.errorLog.addErrorTrace(recordedError, TRACE_PASSTHRU(trace)));
}

//----------------------------------------------------------------

}

#endif
