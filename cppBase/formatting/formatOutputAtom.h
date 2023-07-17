#pragma once

#include "formatting/formatStream.h"
#include "kit/kit.h"

//================================================================
//
// FormatOutputAtom
//
// Interface "output something to the stream".
// Implemented as function+pointer for efficiency.
//
//================================================================

typedef void FormatOutputAtomFunc(const void* value, FormatOutputStream& outputStream);

//----------------------------------------------------------------

struct FormatOutputAtom
{

public:

    const void* value;
    FormatOutputAtomFunc* func;

public:

    sysinline FormatOutputAtom()
        {} // uninitialized

    sysinline FormatOutputAtom(const void* value, FormatOutputAtomFunc* func)
        : value(value), func(func) {}

    //
    // Makes output atom from a value using standard overloaded "formatOutput" function.
    // Attention: stores pointer to the value (not value).
    //

public:

    template <typename Type>
    sysinline FormatOutputAtom(const Type& value)
        : value(&value), func((FormatOutputAtomFunc*) (void*) FormatOutputFunc<Type>::get()) {}

    template <typename Type>
    sysinline void setup(const Type& value)
    {
        this->value = &value;
        this->func = (FormatOutputAtomFunc*) (void*) formatOutput<Type>;
    }

};
