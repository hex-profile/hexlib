#include "foreignExceptReport.h"

#include <exception>

#include "userOutput/printMsg.h"
#include "userOutput/errorLogEx.h"

//================================================================
//
// reportForeignException
//
//================================================================

void reportForeignException(stdPars(ErrorLogExKit)) noexcept
{
    auto exceptionPtr = std::current_exception();

    ////

    try
    {
        std::rethrow_exception(exceptionPtr);
    }
    catch (const ExceptFailure&)
    {
        // Native exception.
    }
    catch (const std::exception& e)
    {
        printMsgTrace(kit.errorLogEx, STR("Standard C++ library exception: %0."), e.what(), msgErr, stdPassThru);
    }
    catch (...)
    {
        printMsgTrace(kit.errorLogEx, STR("Unrecognized external exception."), msgErr, stdPassThru);
    }
}
