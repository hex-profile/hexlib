#pragma once

#include "compileTools/classContext.h"
#include "numbers/int/intBase.h"

#if defined(__linux__)

//================================================================
//
// ErrorLinux
//
//================================================================

class ErrorLinux
{

public:

    operator int () const {return error;}
    CLASS_CONTEXT(ErrorLinux, ((int, error)))

};

//----------------------------------------------------------------

#endif
