#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// LoadNormal
//
//================================================================

struct LoadNormal
{
    template <typename Type>
    static sysinline Type func(const Type* ptr)
        {return *ptr;}
};
