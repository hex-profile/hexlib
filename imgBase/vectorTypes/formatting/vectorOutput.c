#include "vectorTypes/vectorType.h"

#include "formatting/formatStream.h"

//================================================================
//
// outputVectorTypeX2
//
//================================================================

template <typename VectorType>
inline void outputVectorTypeX2(const VectorType& value, FormatOutputStream& outputStream)
{
    outputStream.write(value.x);
    outputStream.write(STR(", "));
    outputStream.write(value.y);
}

template <typename VectorType>
inline void outputVectorTypeX2(const FormatNumber<VectorType>& number, FormatOutputStream& outputStream)
{
    outputStream.write(formatNumber(number.value.x, number.options));
    outputStream.write(STR(", "));
    outputStream.write(formatNumber(number.value.y, number.options));
}

//================================================================
//
// outputVectorTypeX4
//
//================================================================

template <typename VectorType>
inline void outputVectorTypeX4(const VectorType& number, FormatOutputStream& outputStream)
{
    outputStream.write(number.x);
    outputStream.write(STR(", "));
    outputStream.write(number.y);
    outputStream.write(STR(", "));
    outputStream.write(number.z);
    outputStream.write(STR(", "));
    outputStream.write(number.w);
}

template <typename VectorType>
inline void outputVectorTypeX4(const FormatNumber<VectorType>& number, FormatOutputStream& outputStream)
{
    outputStream.write(formatNumber(number.value.x, number.options));
    outputStream.write(STR(", "));
    outputStream.write(formatNumber(number.value.y, number.options));
    outputStream.write(STR(", "));
    outputStream.write(formatNumber(number.value.z, number.options));
    outputStream.write(STR(", "));
    outputStream.write(formatNumber(number.value.w, number.options));
}

//================================================================
//
// formatOutput<VectorType>
//
//================================================================

#define TMP_MACRO(VectorType, func) \
    \
    template <> \
    void formatOutput(const VectorType& value, FormatOutputStream& outputStream) \
        {func(convertFloat32(value), outputStream);} \
    \
    template <> \
    void formatOutput(const FormatNumber<VectorType>& number, FormatOutputStream& outputStream) \
        {func(formatNumber(convertFloat32(number.value), number.options), outputStream);} \

VECTOR_FLOAT_X2_FOREACH(TMP_MACRO, outputVectorTypeX2)
VECTOR_FLOAT_X4_FOREACH(TMP_MACRO, outputVectorTypeX2)

#undef TMP_MACRO

//----------------------------------------------------------------

#define TMP_MACRO(VectorType, func) \
    \
    template <> \
    void formatOutput(const VectorType& value, FormatOutputStream& outputStream) \
        {func(value, outputStream);} \
    \
    template <> \
    void formatOutput(const FormatNumber<VectorType>& number, FormatOutputStream& outputStream) \
        {func(formatNumber(number.value, number.options), outputStream);} \

VECTOR_INT_X2_FOREACH(TMP_MACRO, outputVectorTypeX2)
VECTOR_INT_X4_FOREACH(TMP_MACRO, outputVectorTypeX2)

#undef TMP_MACRO
