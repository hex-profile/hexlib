#pragma once

#include "numbers/int/intBase.h"
#include "storage/adapters/callable.h"
#include "storage/smartPtr.h"

namespace timeoutHelper {

//================================================================
//
// Callback
//
//================================================================

using Callback = Callable<void ()>; // Async!

//================================================================
//
// TimeoutHelper
//
// TimeoutHelper provides asynchronous task execution with a timeout.
//
// Usage:
//
// * Assign a task to the TimeoutHelper by providing a pointer to a callback function
// and a timeout in milliseconds.
//
// * The assigned task will be executed asynchronously after the specified timeout.
//
// * The task can be changed or canceled at any time.
//
// * If a new task is assigned while a previous task is still pending, the previous task will be
//   canceled and replaced with the new one.
//
// * When the TimeoutHelper object is destroyed, any pending task will be canceled and the
//   associated thread will be terminated.
//
// Note:
// - The waiting time in the thread is interruptible, meaning it can be interrupted by a shutdown,
//   task change, or task cancellation.
//
//================================================================

struct TimeoutHelper
{
    static UniquePtr<TimeoutHelper> create() may_throw;
    virtual ~TimeoutHelper() {}

    ////

    virtual void setTask(uint32 waitTimeMilliseconds, Callback& callback) =0;
    virtual void cancelTask() =0;
};

//----------------------------------------------------------------

}

using timeoutHelper::TimeoutHelper;
