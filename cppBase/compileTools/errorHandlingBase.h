#pragma once

//================================================================
//
// Failure
//
// All exceptions are handled regardless of their type,
// only as an indication of failure.
//
// So the only type of internal exception is "Failure".
//
//================================================================

struct Failure {};

//----------------------------------------------------------------

[[noreturn]]
void throwFailure();

//================================================================
//
// ENSURE
//
// Check and fail without user message.
//
//================================================================

#define ENSURE(condition, failReport) \
    (allv(condition) || (throwFailure(), false))

//================================================================
//
// exceptBlock*
//
// Catches everything inside the block.
//
//================================================================

#define exceptBlockBeg \
    try \
    {

#define exceptBlockEnd \
    \
    } \
    catch (...) {}

//----------------------------------------------------------------

#define exceptBlockBegEx(flag) \
    \
    bool flag = false; \
    \
    try \
    { \
        bool& __exceptBlockFlag = flag;
        
#define exceptBlockEndEx \
    \
        __exceptBlockFlag = true; \
    } \
    catch (...) {}

//----------------------------------------------------------------

template <typename Action>
inline bool exceptBlockHelper(const Action& action)
{
    bool ok = false;
    try {action(); ok = true;}
    catch (...) {}
    return ok;
}

#define exceptBlock(action) \
    exceptBlockHelper([&] () {action;})
