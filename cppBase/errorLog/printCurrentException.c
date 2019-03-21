#include "printCurrentException.h"

#include <exception>

#include "userOutput/printMsg.h"

//================================================================
//
// printCurrentException
//
//================================================================

void printCurrentException(MsgLog& log)
{
    using namespace std;

    auto exceptionPtr = current_exception();

    if (!exceptionPtr)
    {
        printMsg(log, STR("No exception"), msgErr);
        return;
    }

    ////

    try
    {
        rethrow_exception(exceptionPtr);
    }
    catch (const exception& e)
    {
        printMsg(log, STR("Standard C++ library exception: %0"), e.what(), msgErr);
        return;
    }
    catch (...)
    {
        printMsg(log, STR("Unrecognized exception"), msgErr);
        return;
    }
}
