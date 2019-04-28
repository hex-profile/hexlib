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
    catch (const Failure&)
    {
        // Native exception.
    }
    catch (const std::exception& e)
    {
        stdDiscard(printMsgTrace(kit.errorLogEx, STR("Standard C++ library exception: %0."), e.what(), msgErr, stdPassThru));
    }
    catch (...)
    {
        stdDiscard(printMsgTrace(kit.errorLogEx, STR("Unrecognized external exception."), msgErr, stdPassThru));
    }
}
