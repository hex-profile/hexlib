#pragma once

#include "storage/opaqueStruct.h"
#include "brood.h"
#include "charType/charArray.h"
#include "kit/kit.h"
#include "numbers/float/floatBase.h"
#include "numbers/int/intBase.h"
#include "stdFunc/traceCallstack.h"

//================================================================
//
// ProfilerScope
//
//================================================================

union ProfilerScope
{
    struct Profiler* profiler;
    OpaqueStruct<24> data;
};

//================================================================
//
// ProfilerNodeLink
//
//================================================================

class ProfilerNodeLink
{

public:

    sysinline bool connected() const {return profilerChild.getParent() != 0;}
    sysinline void disconnect() {profilerChild.disconnect();}

public:

    using InternalReference = OpaqueStruct<8>;
    InternalReference reference;

    BroodChildRole profilerChild;
};

//================================================================
//
// Profiler interface
//
// For efficiency, pointers to functions are used instead of virtual functions.
//
//================================================================

struct Profiler
{
    typedef void Enter(Profiler& profiler, ProfilerScope& scope, TraceLocation location);
    typedef void EnterEx(Profiler& profiler, ProfilerScope& scope, TraceLocation location, uint32 elemCount, const CharArray& userName);
    typedef void Leave(Profiler& profiler, ProfilerScope& scope);

    typedef void GetCurrentNodeLink(Profiler& profiler, ProfilerNodeLink& result);
    typedef void AddDeviceTime(const ProfilerNodeLink& node, float32 deviceTime, float32 overheadTime);

    Enter* const enter;
    EnterEx* const enterEx;
    Leave* const leave;
    GetCurrentNodeLink* const getCurrentNodeLink;
    AddDeviceTime* const addDeviceTime;

    sysinline Profiler(Enter* enter, EnterEx* enterEx, Leave* leave, GetCurrentNodeLink* getCurrentNodeLink, AddDeviceTime* addDeviceTime)
        : enter(enter), enterEx(enterEx), leave(leave), getCurrentNodeLink(getCurrentNodeLink), addDeviceTime(addDeviceTime) {}
};

//================================================================
//
// ProfilerKit
//
//================================================================

KIT_CREATE1(ProfilerKit, Profiler*, profiler);

//================================================================
//
// ProfilerFrame
//
// Enters/leaves profiler on construction/destruction
// bool parameter specifies profiler presence
//
//================================================================

template <bool on>
class ProfilerFrame;


//----------------------------------------------------------------

template <>
class ProfilerFrame<false>
{

public:

    template <typename Kit>
    sysinline ProfilerFrame(const Kit& kit, TraceLocation location) {}

    template <typename Kit>
    sysinline ProfilerFrame(const Kit& kit, TraceLocation location, uint32 elemCount, const CharArray& userName) {}

};

//----------------------------------------------------------------

template <>
class ProfilerFrame<true>
{

public:

    ProfilerScope& scope() {return theScope;}
    const ProfilerScope& scope() const {return theScope;}

public:

    template <typename Kit>
    sysinline ProfilerFrame(const Kit& kit, TraceLocation location)
    {
        Profiler* p = kit.profiler;
        theScope.profiler = p;

        if (p) p->enter(*p, theScope, location);
    }

    template <typename Kit>
    sysinline ProfilerFrame(const Kit& kit, TraceLocation location, uint32 elemCount, const CharArray& userName)
    {
        Profiler* p = kit.profiler;
        theScope.profiler = p;

        if (p) p->enterEx(*p, theScope, location, elemCount, userName);
    }

    sysinline ~ProfilerFrame()
    {
        Profiler* p = theScope.profiler;
        if (p) p->leave(*p, theScope);
    }

private:

    ProfilerScope theScope;

};

//================================================================
//
// Compile-time profiler detection.
//
// We have a profiler if the kit pointer is convertible to ProfilerKit pointer.
//
//================================================================

template <typename Kit>
struct ProfilerKitPresent
{
    using FalseType = char;
    struct TrueType {char data[2];};
    COMPILE_ASSERT(sizeof(FalseType) != sizeof(TrueType));

    static Kit getKit;

    static FalseType detectProfiler(...);
    static TrueType detectProfiler(const ProfilerKit*);

    static constexpr bool value = sizeof(detectProfiler(&getKit)) == sizeof(TrueType);
};

////

#define PROFILER__KIT_PRESENT(kit) \
    ProfilerKitPresent<decltype(kit)>::value

////

#define PROFILER__FRAME(kit) \
    ProfilerFrame<PROFILER__KIT_PRESENT(kit)>

//================================================================
//
// PROFILER_FRAME_ENTER
//
// Macros for entering/leaving profiler block
//
//================================================================

#define PROFILER_FRAME_ENTER(kit, location) \
    PROFILER__FRAME(kit) PREP_PASTE(__profilerFrame, __LINE__)(kit, location)

#define PROFILER_FRAME_ENTER_EX(kit, location, elemCount, userName) \
    PROFILER__FRAME(kit) PREP_PASTE(__profilerFrame, __LINE__)(kit, location, elemCount, userName)

#define PROFILER_FRAME_TEMPORARY(kit, location) \
    PROFILER__FRAME(kit)(kit, location)
