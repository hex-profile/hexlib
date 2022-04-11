#include "types/movement3d/movement3d.h"

#include "formatting/formatStream.h"
#include "userOutput/formatMsg.h"

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
    auto& p = number.value;

    formatMsg
    (
        outputStream,
        STR("{\"q\": {\"w\": %, \"x\": %, \"y\": %, \"z\": %}, \"p\": {\"x\": %, \"y\": %, \"z\": %}}"),
        formatNumber(p.rotation.W, number.options),
        formatNumber(p.rotation.X, number.options),
        formatNumber(p.rotation.Y, number.options),
        formatNumber(p.rotation.Z, number.options),
        formatNumber(p.translation.X, number.options),
        formatNumber(p.translation.Y, number.options),
        formatNumber(p.translation.Z, number.options)
    );

    //outputStream.write(STR("{"));
    //formatOutput(formatNumber(number.value.rotation, number.options), outputStream);
    //outputStream.write(STR("}, {"));
    //formatOutput(formatNumber(number.value.translation, number.options), outputStream);
    //outputStream.write(STR("}"));
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
