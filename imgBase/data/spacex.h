#pragma once

#include "numbers/safeint32/safeint32.h"
#include "data/space.h"

#ifdef SAFEINT32_SUPPORT
#define SPACEX_SUPPORT

//================================================================
//
// Spacex
//
// The same as Space type, but with built-in error control.
//
//================================================================

using Spacex = safeint32::Type;

COMPILE_ASSERT(TYPE_EQUAL(TYPE_MAKE_UNCONTROLLED(Spacex), Space));

//================================================================
//
// spacex
//
// Import (Checked)
//
//================================================================

template <typename Src>
inline typename ConvertResult<Src, Spacex>::T spacex(const Src& value)
    {return convertExact<Spacex>(value);}

//================================================================
//
// spacexLiteral
//
// Import (Unchecked)
//
//================================================================

template <typename Src>
inline typename ConvertResult<Src, Spacex>::T spacexLiteral(const Src& value)
    {return convertExactUnchecked<Spacex>(value);}

//================================================================
//
// spacexNan
//
//================================================================

inline Spacex spacexNan()
    {return nanOf<Spacex>();}

//----------------------------------------------------------------

#endif
