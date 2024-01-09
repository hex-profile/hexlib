#include "convertExceptions.h"

#include <exception>

#include "userOutput/printMsg.h"
#include "userOutput/printMsgTrace.h"

//================================================================
//
// printExternalExceptions
//
//================================================================

void printExternalExceptions(stdPars(MsgLogExKit)) noexcept
{
    try
    {

        auto exceptionPtr = std::current_exception();

        ////

        try
        {
            std::rethrow_exception(exceptionPtr);
        }

    #if HEXLIB_ERROR_HANDLING == 1

        catch (const ExceptFailure&)
        {
            // The native exception.
        }

    #endif

        catch (const CharType* msg)
        {
            errorBlock(printMsgTrace(STR("%0."), msg, msgErr, stdPassThruNc));
        }

        catch (const std::bad_alloc& e)
        {
            errorBlock(printMsgTrace(STR("Standard C++ library error: Memory allocation failed."), e.what(), msgErr, stdPassThruNc));
        }

        catch (const std::exception& e)
        {
            errorBlock(printMsgTrace(STR("Standard C++ library error: %."), e.what(), msgErr, stdPassThruNc));
        }

        catch (...)
        {
            errorBlock(printMsgTrace(STR("Unrecognized external exception."), msgErr, stdPassThruNc));
        }

    }
    catch (...)
    {
    }
}
