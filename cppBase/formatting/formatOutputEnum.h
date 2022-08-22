#pragma once

#include "formatting/formatStream.h"
#include "prepTools/prepFor.h"
#include "prepTools/prepSeq.h"
#include "prepTools/prepArg.h"

//================================================================
//
// FORMAT_OUTPUT_ENUM
//
// Makes formatOutput function for an enumerated type.
//
//================================================================

#define FORMAT_OUTPUT_ENUM(EnumType, seq) \
    \
    template <> \
    void formatOutput(const EnumType& value, FormatOutputStream& outputStream) \
    { \
        if (0) ; \
        PREP_FOR(PREP_SEQ_SIZE(seq), FORMAT_OUTPUT_ENUM_ITER, seq) \
        else outputStream << int(value); \
    }

#define FORMAT_OUTPUT_ENUM_ITER(n, seq) \
    FORMAT_OUTPUT_ENUM_ITER2(PREP_SEQ_ELEM(n, seq))

#define FORMAT_OUTPUT_ENUM_ITER2(pair) \
    FORMAT_OUTPUT_ENUM_ITER3(PREP_ARG2_0 pair, PREP_ARG2_1 pair)

#define FORMAT_OUTPUT_ENUM_ITER3(enumValue, stringLiteral) \
    else if (value == enumValue) outputStream << STR(stringLiteral);

//================================================================
//
// FORMAT_OUTPUT_ENUM_SIMPLE
//
// Makes formatOutput function for an enumerated type.
//
//================================================================

#define FORMAT_OUTPUT_ENUM_SIMPLE(EnumType, seq) \
    \
    template <> \
    void formatOutput(const EnumType& value, FormatOutputStream& outputStream) \
    { \
        if (0) ; \
        PREP_FOR(PREP_SEQ_SIZE(seq), FORMAT_OUTPUT_ENUM_SIMPLE_ITER, seq) \
        else outputStream << int(value); \
    }

#define FORMAT_OUTPUT_ENUM_SIMPLE_ITER(n, seq) \
    FORMAT_OUTPUT_ENUM_SIMPLE_ITER2(PREP_SEQ_ELEM(n, seq))

#define FORMAT_OUTPUT_ENUM_SIMPLE_ITER2(enumValue) \
    else if (value == enumValue) outputStream << STR(PREP_STRINGIZE(enumValue));
