#include "foreignErrorBlock.h"

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
            printMsgTrace(kit.errorLogEx, STR("Unrecognized foreign exception."), msgErr, stdPassThru);
        }

    }
    catch (...)
    {
    }
}
