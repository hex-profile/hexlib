#if defined(__linux__)

#include "errorLinux.h"

#include "errorLog/debugBreak.h"
#include "formatting/formatModifiers.h"
#include "formatting/formatStream.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// ErrorLinux
//
//================================================================

template <>
void formatOutput(const ErrorLinux& value, FormatOutputStream& outputStream)
{
    outputStream << STR("Error ") << value.get();

    ////

    constexpr int bufferSize = 256;
    char bufferArray[bufferSize];
    char* errorMsg = strerror_r(value.get(), bufferArray, bufferSize);

    outputStream << STR(". ") << charArrayFromPtr(errorMsg);
}

//----------------------------------------------------------------

#endif
