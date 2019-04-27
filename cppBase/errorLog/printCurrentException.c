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
    auto exceptionPtr = std::current_exception();

    if (!exceptionPtr)
    {
        printMsg(log, STR("No exception"), msgErr);
        return;
    }

    ////

    try
    {
        std::rethrow_exception(exceptionPtr);
    }
    catch (const std::exception& e)
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
