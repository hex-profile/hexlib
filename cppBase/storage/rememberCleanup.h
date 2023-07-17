#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// REMEMBER_CLEANUP
//
// Remembers to perform a cleanup.
// The user specifies a cleanup statement, the type and name of a variable to remember.
//
// For example:
//
// REMEMBER_CLEANUP(free(ptr));
//
// This results in calling free(ptr) at the end of the scope.
// The cleanup is always executed, even on abnormal control flow,
// like returning an error code.
//
// If you need to cancel cleanup sometimes, use REMEMBER_CLEANUP_EX.
//
//================================================================

//================================================================
//
// RC__PASTE
//
//================================================================

#define RC__PASTE_AUX(A, B) \
    A ## B

#define RC__PASTE(A, B) \
    RC__PASTE_AUX(A, B)

//================================================================
//
// RememberCleanup
//
// Modern C11 cleanup (move not implemented yet)
//
//================================================================

template <typename Action>
class RememberCleanup
{

public:

    sysinline RememberCleanup(const Action& action)
        : action(action) {}

    sysinline ~RememberCleanup()
        {action();}

private:

    Action action;

};

//----------------------------------------------------------------

template <typename Action>
sysinline RememberCleanup<Action> rememberCleanup(const Action& action)
    {return RememberCleanup<Action>(action);}

//----------------------------------------------------------------

#define REMEMBER_CLEANUP(action) \
    auto RC__PASTE(__cleanup, __LINE__) = rememberCleanup([&] () {action;})

//----------------------------------------------------------------

#define REMEMBER_COUNTING_CLEANUP(action) \
    REMEMBER_CLEANUP(if (!kit.dataProcessing) {action;})

//================================================================
//
// RememberCleanupEx
//
//================================================================

template <typename Action>
class RememberCleanupEx
{

public:

    sysinline RememberCleanupEx(const Action& action)
        : action(action) {}

    sysinline ~RememberCleanupEx()
        {if (active) action();}

    sysinline void cancel()
        {active = false;}

    sysinline void activate()
        {active = true;}

    sysinline void setActive(bool active)
        {this->active = active;}

private:

    bool active = true;
    Action action;

};

//----------------------------------------------------------------

template <typename Action>
sysinline RememberCleanupEx<Action> rememberCleanupEx(const Action& action)
    {return RememberCleanupEx<Action>(action);}

//----------------------------------------------------------------

#define REMEMBER_CLEANUP_EX(name, action) \
    auto name = rememberCleanupEx([&] () {action;})
