#pragma once

#include "compileTools/classContext.h"
#include "numbers/int/intBase.h"

#if defined(_WIN32)

//================================================================
//
// ErrorWin32
//
//================================================================

class ErrorWin32
{

public:

    uint32 get() const {return error;}
    CLASS_CONTEXT(ErrorWin32, ((uint32, error)))

};

//----------------------------------------------------------------

#endif
