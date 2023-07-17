#include "printSignalKey.h"

#include "formatting/formatStream.h"

//================================================================
//
// formatOutput<PrintSignalKey>
//
//================================================================

template <>
void formatOutput(const PrintSignalKey& value, FormatOutputStream& outputStream)
{
    auto toStream = cfgOutputString | [&] (auto ptr, auto size)
    {
        outputStream.write(ptr, size);
        return true;
    };

    value.signal.getKey(toStream);
}
