#include "convertAllExceptions.h"

#include <exception>

#include "userOutput/printMsg.h"
#include "userOutput/errorLogEx.h"

//================================================================
//
// printExternalExceptions
//
//================================================================

void printExternalExceptions(stdPars(ErrorLogExKit)) noexcept
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
            printMsgTrace(kit.errorLogEx, STR("%0."), msg, msgErr, stdPassThru);
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
    catch (...)
    {
    }
}
