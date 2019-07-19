#include "types/movement3d/movement3d.h"

#include "formatting/formatStream.h"

//================================================================
//
// output
//
//================================================================

template <typename Float>
inline void output(const Movement3D<Float>& value, FormatOutputStream& outputStream)
{
    outputStream.write(STR("{"));
    formatOutput(value.rotation, outputStream);
    outputStream.write(STR("}, {"));
    formatOutput(value.translation, outputStream);
    outputStream.write(STR("}"));
}

//================================================================
//
// output
//
//================================================================

template <typename Float>
inline void output(const FormatNumber<Movement3D<Float>>& number, FormatOutputStream& outputStream)
{
    outputStream.write(STR("{"));
    formatOutput(formatNumber(number.value.rotation, number.options), outputStream);
    outputStream.write(STR("}, {"));
    formatOutput(formatNumber(number.value.translation, number.options), outputStream);
    outputStream.write(STR("}"));
}

//================================================================
//
// formatOutput<Movement3D<Float>>
//
//================================================================

#define TMP_MACRO(Float, o) \
    \
    template <> \
    void formatOutput(const Movement3D<Float>& value, FormatOutputStream& outputStream) \
        {output(value, outputStream);} \
    \
    template <> \
    void formatOutput(const FormatNumber<Movement3D<Float>>& value, FormatOutputStream& outputStream) \
        {output(value, outputStream);}

BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
