#pragma once

#include "errorLog/errorLogKit.h"
#include "numbers/float/floatBase.h"
#include "stdFunc/profiler.h"
#include "stdFunc/stdFunc.h"
#include "dataAlloc/arrayMemory.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "profilerShell/profiler/profilerTimer.h"
#include "dataAlloc/arrayObjectMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "allocation/mallocKit.h"

//================================================================
//
// ProfilerDeviceInterface
//
//================================================================

using ProfilerDeviceKit = KitCombine<GpuCurrentStreamKit, GpuStreamWaitingKit>;

//================================================================
//
// ProfilerNode
// ProfilerScopeEx
//
//================================================================

struct ProfilerNode;
struct ProfilerScopeEx;

//================================================================
//
// ProfilerImpl
//
//================================================================

class ProfilerImpl
{

public:

    using AllocKit = KitCombine<ErrorLogKit, MallocKit>;

public:

    stdbool realloc(Space capacity, stdPars(AllocKit));
    void dealloc();

public:

    inline void setDeviceControl(const ProfilerDeviceKit* deviceControl)
        {this->deviceControl = deviceControl;}

public:

    bool checkResetScope();
    void resetMemory();

    bool created() const {return nodePool.size() >= 1;}

public:

    inline float32 divTicksPerSec()
        {return timer.divTicksPerSec();}

    inline ProfilerNode* getRootNode()
    {
        ARRAY_EXPOSE(nodePool);
        return (nodePoolSize >= 1) ? unsafePtr(nodePoolPtr, 1) : 0;
    }

    bool capacityExceeded() const
    {
        return (nodeCount == nodePool.size());
    }

public:

    inline void enter(ProfilerScopeEx& scope, TraceLocation location, uint32 elemCount, const CharArray& userName);
    inline void leave(ProfilerScopeEx& scope);
    inline void getCurrentNodeLink(ProfilerNodeLink& result);
    static inline void addDeviceTime(ProfilerNode* node, float32 deviceTime, float32 overheadTime);

private:

    bool gpuCoverageInitialized = false;

    // Timer
    ProfilerTimerEx timer;

    // Device interop
    const ProfilerDeviceKit* deviceControl = 0;

    // The number of used nodes
    Space nodeCount = 0;

    // Memory pool for fast allocation
    ArrayMemory<ProfilerNode> nodePool;
    BroodParentRole nodeRefOwner;

    // If currentScope == 0, then nodeCount == nodePoolSize (out of memory)
    ProfilerNode* currentScope = 0;

};

//================================================================
//
// ProfilerThunk
//
//================================================================

class ProfilerThunk : public Profiler
{

public:

    inline ProfilerThunk(ProfilerImpl& impl)
        :
        Profiler(enterFunc, enterExFunc, leaveFunc, getCurrentNodeLinkFunc, addDeviceTimeFunc),
        impl(impl)
    {
    }

public:

    static void enterFunc(Profiler& profiler, ProfilerScope& scope, TraceLocation location);
    static void enterExFunc(Profiler& profiler, ProfilerScope& scope, TraceLocation location, uint32 elemCount, const CharArray& userName);
    static void leaveFunc(Profiler& profiler, ProfilerScope& scope);
    static void getCurrentNodeLinkFunc(Profiler& profiler, ProfilerNodeLink& result);
    static void addDeviceTimeFunc(const ProfilerNodeLink& node, float32 deviceTime, float32 overheadTime);

private:

    ProfilerImpl& impl;

};
