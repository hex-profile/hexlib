#include "vectorTypes/vectorType.h"

#include "formatting/formatStream.h"

//================================================================
//
// outputVectorTypeX2
//
//================================================================

template <typename VectorType>
sysinline void outputVectorTypeX2(const VectorType& value, FormatOutputStream& outputStream)
{
    outputStream
        << value.x << STR(", ")
        << value.y;
}

template <typename VectorType>
sysinline void outputVectorTypeX2(const FormatNumber<VectorType>& number, FormatOutputStream& outputStream)
{
    outputStream
        << formatNumber(number.value.x, number) << STR(", ")
        << formatNumber(number.value.y, number);
}

//================================================================
//
// outputVectorTypeX4
//
//================================================================

template <typename VectorType>
sysinline void outputVectorTypeX4(const VectorType& number, FormatOutputStream& outputStream)
{
    outputStream
        << number.x << STR(", ")
        << number.y << STR(", ")
        << number.z << STR(", ")
        << number.w;
}

template <typename VectorType>
sysinline void outputVectorTypeX4(const FormatNumber<VectorType>& number, FormatOutputStream& outputStream)
{
    outputStream
        << formatNumber(number.value.x, number) << STR(", ")
        << formatNumber(number.value.y, number) << STR(", ")
        << formatNumber(number.value.z, number) << STR(", ")
        << formatNumber(number.value.w, number);
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
        {func(formatNumber(convertFloat32(number.value), number), outputStream);} \

VECTOR_FLOAT_X2_FOREACH(TMP_MACRO, outputVectorTypeX2)
VECTOR_FLOAT_X4_FOREACH(TMP_MACRO, outputVectorTypeX4)

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
        {func(formatNumber(number.value, number), outputStream);} \

VECTOR_INT_X2_FOREACH(TMP_MACRO, outputVectorTypeX2)
VECTOR_INT_X4_FOREACH(TMP_MACRO, outputVectorTypeX4)

#undef TMP_MACRO
