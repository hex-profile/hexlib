#include "stlString.h"

#include "formatting/formatStream.h"

//================================================================
//
// formatOutput
//
//================================================================

template <>
void formatOutput(const StlString& value, FormatOutputStream& stream)
{
    stream.write(value.data(), value.size());
}
