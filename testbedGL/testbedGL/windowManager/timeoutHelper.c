#include "timeoutHelper.h"

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "setThreadName/setThreadName.h"
#include "compileTools/blockExceptionsSilent.h"

namespace timeoutHelper {

using namespace std;

//================================================================
//
// TimeoutHelperImpl
//
//================================================================

struct TimeoutHelperImpl : public TimeoutHelper
{

    //----------------------------------------------------------------
    //
    // Init/deinit.
    //
    //----------------------------------------------------------------

    TimeoutHelperImpl() may_throw
    {
        stream = thread(&TimeoutHelperImpl::threadFunction, this);
    }

    ~TimeoutHelperImpl()
    {
        {
            unique_lock<mutex> guard(lock);
            isShutdown = true;
        }

        condition.notify_one();

        stream.join();
    }

    //----------------------------------------------------------------
    //
    // setTask
    //
    //----------------------------------------------------------------

    void setTask(uint32 waitTimeMilliseconds, Callback& callback)
    {
        {
            unique_lock<mutex> guard(lock);

            taskPtr = &callback;
            taskTimeout = waitTimeMilliseconds;
            taskChange = true;
        }

        condition.notify_one();
    }

    //----------------------------------------------------------------
    //
    // cancelTask
    //
    //----------------------------------------------------------------

    void cancelTask()
    {
        {
            unique_lock<mutex> guard(lock);

            taskPtr = nullptr;
            taskTimeout = 0;
            taskChange = true;
        }

        condition.notify_one();
    }

    //----------------------------------------------------------------
    //
    // threadFunction
    //
    //----------------------------------------------------------------

    void threadFunction();

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    mutex lock;
    condition_variable condition;

    ////

    bool isShutdown = false;

    Callback* taskPtr = nullptr;
    uint32 taskTimeout = 0;

    bool taskChange = false;

    ////

    thread stream;

};

////

UniquePtr<TimeoutHelper> TimeoutHelper::create()
    {return makeUnique<TimeoutHelperImpl>();}

//================================================================
//
// TimeoutHelperImpl::threadFunction
//
//================================================================

void TimeoutHelperImpl::threadFunction()
{
    setThreadName(STR("~TimeoutHelper"));

    for (;;)
    {

        Callback* execPtr = nullptr;

        auto taskChangeOrShutdown = [&]() {return isShutdown || taskChange;};

        //----------------------------------------------------------------
        //
        // Wait.
        //
        //----------------------------------------------------------------

        {
            unique_lock<mutex> guard(lock);

            ////

            while (!execPtr)
            {
                if (!taskPtr)
                    condition.wait(guard, taskChangeOrShutdown);
                else
                {
                    auto duration = chrono::milliseconds(taskTimeout);

                    if (!condition.wait_for(guard, duration, taskChangeOrShutdown))
                    {
                        execPtr = taskPtr;
                        taskPtr = nullptr;
                        taskTimeout = 0;
                    }
                }

                taskChange = false;

                if (isShutdown)
                    return;
            }
        }

        //----------------------------------------------------------------
        //
        // Exec.
        //
        //----------------------------------------------------------------

        blockExceptBegin;
        (*execPtr)();
        blockExceptEndIgnore;

    }
}

//----------------------------------------------------------------

}
