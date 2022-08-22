#include "types/movement3d/movement3d.h"

#include "formatting/formatStream.h"
#include "userOutput/formatMsg.h"

//================================================================
//
// output
//
//================================================================

template <typename Float>
inline void output(const FormatNumber<Movement3D<Float>>& number, FormatOutputStream& outputStream)
{
    auto& p = number.value;

    formatMsg
    (
        outputStream,
        STR("{\"q\": {\"w\": %, \"x\": %, \"y\": %, \"z\": %}, \"p\": {\"x\": %, \"y\": %, \"z\": %}}"),
        formatNumber(p.rotation.W, number),
        formatNumber(p.rotation.X, number),
        formatNumber(p.rotation.Y, number),
        formatNumber(p.rotation.Z, number),
        formatNumber(p.translation.X, number),
        formatNumber(p.translation.Y, number),
        formatNumber(p.translation.Z, number)
    );
}

//================================================================
//
// formatOutput<Movement3D<Float>>
//
//================================================================

#define TMP_MACRO(Float, o) \
    \
    template <> \
    void formatOutput(const FormatNumber<Movement3D<Float>>& value, FormatOutputStream& outputStream) \
        {output(value, outputStream);}

BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
