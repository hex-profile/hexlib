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
    typedef void Enter(Profiler& profiler, ProfilerScope& scope, TraceLocation location, uint32 elemCount, const CharArray& userName);
    typedef void Leave(Profiler& profiler, ProfilerScope& scope);

    typedef void GetCurrentNodeLink(Profiler& profiler, ProfilerNodeLink& result);
    typedef void AddDeviceTime(const ProfilerNodeLink& node, float32 deviceTime, float32 overheadTime);

    Enter* enter;
    Leave* leave;
    GetCurrentNodeLink* getCurrentNodeLink;
    AddDeviceTime* addDeviceTime;

    sysinline Profiler(Enter* enter, Leave* leave, GetCurrentNodeLink* getCurrentNodeLink, AddDeviceTime* addDeviceTime)
        : enter(enter), leave(leave), getCurrentNodeLink(getCurrentNodeLink), addDeviceTime(addDeviceTime) {}
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
class ProfilerFrame<true>
{

public:

    ProfilerScope& scope() {return theScope;}
    const ProfilerScope& scope() const {return theScope;}

public:

    template <typename Kit>
    sysinline ProfilerFrame(const Kit& kit, TraceLocation location, uint32 elemCount, const CharArray& userName)
    {
        Profiler* p = kit.profiler;
        theScope.profiler = p;
        if (p) p->enter(*p, theScope, location, elemCount, userName);
    }

    sysinline ~ProfilerFrame()
    {
        Profiler* p = theScope.profiler;
        if (p) p->leave(*p, theScope);
    }

private:

    ProfilerScope theScope;

};

//----------------------------------------------------------------

template <>
class ProfilerFrame<false>
{

public:

    template <typename Kit>
    sysinline ProfilerFrame(const Kit& kit, TraceLocation location, uint32 elemCount, const CharArray& userName) {}

};

//================================================================
//
// Compile-time profiler detection
//
// We have profiler if kit pointer is castable to ProfilerKit pointer.
//
//================================================================

using ProfilerFalseType = char;
struct ProfilerTrueType {char data[2];};

ProfilerFalseType profilerDetectKit(...);
ProfilerTrueType profilerDetectKit(const ProfilerKit*);

#define PROFILER__KIT_PRESENT(kit) \
    (sizeof(profilerDetectKit(&kit)) == sizeof(ProfilerTrueType))

#define PROFILER__FRAME(kit) \
    ProfilerFrame<PROFILER__KIT_PRESENT(kit)>

//================================================================
//
// PROFILER_BLOCK*
//
// Macros for entering/leaving profiler block
//
//================================================================

#define PROFILER_STD_FRAME _profilerFrame

#define PROFILER_SCOPE_EX(location, elemCount, userName) \
    PROFILER__FRAME(kit) PROFILER_STD_FRAME(kit, location, elemCount, userName)
