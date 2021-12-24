#pragma once

#include "charType/charArray.h"
#include "charType/strUtils.h"
#include "numbers/float/floatType.h"
#include "parseTools/readTools.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// cmdArgRead
//
//================================================================

inline bool cmdArgRead(const CharType*& paramVar, const CharType* cmdArg, const CharType* prefixPtr)
{
    size_t prefixSize = strLen(prefixPtr);

    bool recognized = false;

    if (prefixSize <= strlen(cmdArg) && memcmp(prefixPtr, cmdArg, prefixSize * sizeof(CharType)) == 0)
    {
        paramVar = cmdArg + prefixSize;
        recognized = true;
    }

    return recognized;
}

//================================================================
//
// cmdArgGetFloat
//
//================================================================

inline void cmdArgGetFloat(const CharType* cstr, float32& result)
{
    REMEMBER_CLEANUP_EX(invalidate, result = float32Nan());

    auto str = charArrayFromPtr(cstr);

    auto ptr = str.ptr;
    auto end = str.ptr + str.size;

    ensurev(readFloatApprox(ptr, end, result));
    ensurev(ptr == end);

    invalidate.cancel();
}

//================================================================
//
// cmdArgGetInt
//
//================================================================

template <typename Int>
inline void cmdArgGetInt(const CharType* cstr, Int errorValue, Int& result)
{
    REMEMBER_CLEANUP_EX(invalidate, result = errorValue);

    auto str = charArrayFromPtr(cstr);

    auto ptr = str.ptr;
    auto end = str.ptr + str.size;

    ensurev(readInt(ptr, end, result));
    ensurev(ptr == end);

    invalidate.cancel();
}
