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
            printMsgTrace(kit.msgLogEx, STR("%0."), msg, msgErr, stdPassThru);
        }

        catch (const std::bad_alloc& e)
        {
            printMsgTrace(kit.msgLogEx, STR("Standard C++ library error: Memory allocation failed."), e.what(), msgErr, stdPassThru);
        }

        catch (const std::exception& e)
        {
            printMsgTrace(kit.msgLogEx, STR("Standard C++ library error: %."), e.what(), msgErr, stdPassThru);
        }

        catch (...)
        {
            printMsgTrace(kit.msgLogEx, STR("Unrecognized external exception."), msgErr, stdPassThru);
        }

    }
    catch (...)
    {
    }
}
