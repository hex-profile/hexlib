#pragma once

#include "compileTools/compileTools.h"
#include "numbers/int/intBase.h"
#include "storage/addrSpace.h"

namespace hashTools {

//================================================================
//
// Constants.
//
//================================================================

constexpr uint32 fnvBasis = 2166136261U;
constexpr uint32 fnvPrime = 16777619U;

//================================================================
//
// hashBytes
//
// Warning: The hash is 32 bit.
//
//================================================================

sysinline uint32 hashBytes(const void* ptr, size_t size, uint32 startValue = fnvBasis)
{
	uint32 result = startValue;

    const Byte* bytePtr = (const Byte*) ptr;

	for_count (i, size)
    {
		result ^= uint32(bytePtr[i]);
		result *= fnvPrime;
	}

	return result;
}

//----------------------------------------------------------------

}

using hashTools::hashBytes;

