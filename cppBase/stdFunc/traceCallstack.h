#pragma once

#include "charType/charType.h"
#include "prepTools/prepBase.h"

//================================================================
//
// Call-stack trace support.
//
// A back-linked list of program locations is being maintained.
//
// When new program scope is entered (on function call),
// a new trace scope structure is created, storing the source code location
// and the pointer to the previous scope.
//
// In this version, the source code location is C string literal.
//
// Two implementations are possible:
//
// * On entering new scope, new trace scope instance is passed as a const reference
// to a newly created structure ("aggregated" implementation).
//
// * On entering new scope, the pointer to the old scope instance and source location
// are passed separately ("separate" implementation). In this case, the called function
// should reassemble trace scope at the beginning of the function.
//
// The separate implementation is usually more efficient, but it can't be used
// in constructor initialization lists.
//
//================================================================

//================================================================
//
// TraceLocation
//
//================================================================

using TraceLocation = const CharType*;

//================================================================
//
// TRACE_AUTO_LOCATION
//
//================================================================

#define TRACE_AUTO_LOCATION \
    CT(__FILE__) CT("(") CT(TRACE_STRINGIZE(__LINE__)) CT(")")

#define TRACE_AUTO_LOCATION_MSG(msg) \
    TRACE_AUTO_LOCATION CT(": ") CT(msg)

#define TRACE_STRINGIZE_AUX(X) \
    #X

#define TRACE_STRINGIZE(X) \
    TRACE_STRINGIZE_AUX(X)

//================================================================
//
// TraceScope
//
// Back-linked list of callstack nodes.
//
// Null-terminated strings are used instead of CharArray,
// because node linking happens VERY often, but actual usage is very rare.
//
//================================================================

class TraceScope
{

public:

    TraceLocation const location;
    const TraceScope* const prev;

public:

    inline TraceScope(TraceLocation location, const TraceScope* prev)
        : location(location),  prev(prev) {}

    inline TraceScope(TraceLocation location)
        : location(location), prev(0) {}

};

//================================================================
//
// TRACE_* common macros
//
//================================================================

#define TRACE_ROOT_STD \
    TRACE_ROOT(stdTraceName, TRACE_AUTO_LOCATION)

#define TRACE_SCOPE(trace) \
    PREP_PASTE(trace, Scope)

//================================================================
//
// TRACE_REGISTER_FRIENDLY_IMPL
//
//================================================================

#define TRACE_REGISTER_FRIENDLY_IMPL 0

//================================================================
//
// TRACE_* macros
//
// Register-friendly version.
//
//================================================================

#if TRACE_REGISTER_FRIENDLY_IMPL

////

#define TRACE_PARAMS(trace) \
    const TraceScope* PREP_PASTE(trace, Prev), \
    const TraceLocation PREP_PASTE(trace, Location)

#define TRACE_REASSEMBLE(trace) \
    const TraceScope PREP_PASTE(trace, Scope)(PREP_PASTE(trace, Location), PREP_PASTE(trace, Prev));

#define TRACE_ENTER(trace, location) \
    const TraceScope* PREP_PASTE(trace, Prev) = &PREP_PASTE(trace, Scope); \
    const TraceLocation PREP_PASTE(trace, Location) = (location); \
    TRACE_REASSEMBLE(trace)

#define TRACE_PASS(trace, location) \
    &PREP_PASTE(trace, Scope), (location)

#define TRACE_PASSTHRU(trace) \
    PREP_PASTE(trace, Prev), PREP_PASTE(trace, Location)

#define TRACE_ROOT(trace, location) \
    const TraceScope* PREP_PASTE(trace, Prev) = 0; \
    const TraceLocation PREP_PASTE(trace, Location) = (location); \
    const TraceScope PREP_PASTE(trace, Scope)(PREP_PASTE(trace, Location), PREP_PASTE(trace, Prev))

////

#endif // TRACE_REGISTER_FRIENDLY_IMPL

//================================================================
//
// TRACE_* macros
//
// General-purpose version.
//
//================================================================

#if !TRACE_REGISTER_FRIENDLY_IMPL

////

//
// TRACE_PARAMS: Passing the trace scope by value instead of reference is a bit better for normal calls (at least on MSVC)
// and a bit worse for pass-thru calls, but there are much more normal calls in application code.
//

#define TRACE_PARAMS(trace) \
    const TraceScope TRACE_SCOPE(trace)

#define TRACE_REASSEMBLE(trace)

#define TRACE_ENTER(trace, location) \
    const TraceScope* PREP_PASTE(trace, _Prev) = &TRACE_SCOPE(trace); \
    const TraceScope TRACE_SCOPE(trace)(location, PREP_PASTE(trace, _Prev))

#define TRACE_PASS(trace, location) \
    TraceScope(location, &TRACE_SCOPE(trace))

#define TRACE_PASSTHRU(trace) \
    TRACE_SCOPE(trace)

#define TRACE_ROOT(trace, location) \
    const TraceScope TRACE_SCOPE(trace)(location)

////

#endif // !TRACE_REGISTER_FRIENDLY_IMPL
