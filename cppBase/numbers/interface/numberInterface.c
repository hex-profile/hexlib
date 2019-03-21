#include "numberInterface.h"

#include "formatting/formatStream.h"

//================================================================
//
// formatOutput
//
//================================================================

void formatOutput(const Rounding& rounding, FormatOutputStream& stream)
{
    stream.write
    (
        rounding == RoundDown ? STR("RoundDown") :
        rounding == RoundUp ? STR("RoundUp") :
        rounding == RoundNearest ? STR("RoundNearest") :
        rounding == RoundExact ? STR("RoundExact") :
        STR("")
    );
}
