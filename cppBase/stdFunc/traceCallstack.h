#pragma once

#include "charType/charType.h"
#include "prepTools/prepBase.h"

//================================================================
//
// Call-stack trace.
//
// Contains a back-linked list of program locations.
//
// When new program scope is entered (on function call),
// a new trace scope structure is created, storing the source code location
// and the pointer to the previous scope.
//
// In this version, the source code location is C string literal.
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
// TRACE_* macros
//
// General-purpose version.
//
//================================================================

//
// TRACE_PARAMS: Passing the trace scope by value instead of reference is a bit better for normal calls (at least on MSVC)
// and a bit worse for pass-thru calls, but there are much more normal calls in application code.
//

#define TRACE_PARAMS(trace) \
    TraceScope trace

#define TRACE_ENTER(trace, location) \
    const TraceScope* PREP_PASTE(trace, _Prev) = &trace; \
    const TraceScope trace(location, PREP_PASTE(trace, _Prev))

#define TRACE_PASS(trace, location) \
    TraceScope(location, &trace)

#define TRACE_PASSTHRU(trace) \
    trace

#define TRACE_ROOT(trace, location) \
    const TraceScope trace(location)
