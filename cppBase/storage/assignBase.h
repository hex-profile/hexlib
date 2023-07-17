#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// assignBase
//
//================================================================

template <typename Src, typename Dst>
sysinline void assignBase(Dst& dst, const Src& src)
{
    Src* dstBase = &dst;
    *dstBase = src;
}
