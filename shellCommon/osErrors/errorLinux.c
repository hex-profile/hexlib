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
    outputStream.write(STR("Error "));
    outputStream.write(value);

    ////

    constexpr int bufferSize = 256;
    char bufferArray[bufferSize];
    char* errorMsg = strerror_r(int(value), bufferArray, bufferSize);

    outputStream.write(STR(". "));
    outputStream.write(charArrayFromPtr(errorMsg));
}

//----------------------------------------------------------------

#endif
