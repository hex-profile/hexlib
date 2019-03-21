#include "safeint32_generic.h"

//================================================================
//
// x86
//
//================================================================

COMPILE_ASSERT(sizeof(safeint32::Type) == 4);

//================================================================
//
// operator /
//
//================================================================

safeint32::Type operator /(const safeint32::Type& X, const safeint32::Type& Y)
{
    safeint32::Base Xv = INT32C__ACCESS(X);
    safeint32::Base Yv = INT32C__ACCESS(Y);

    safeint32::Base result = INT32C__INDEFINITE;

    if (Xv != INT32C__INDEFINITE && Yv != INT32C__INDEFINITE)
    {
        if (Yv != 0)
        {
            // dividing by any number (except zero) results in a valid range value
            COMPILE_ASSERT(safeint32::rangeIsSymmetric);

            result = Xv / Yv;
        }
    }

    return INT32C__CREATE(result);
}

//================================================================
//
// operator %
//
//================================================================

bool builtinRemTest()
{
    bool ok = true;

    for (int X = -10; X <= 10; ++X)
    {
        for (int Y = -10; Y <= 10; ++Y)
        {  
            if (Y != 0)
            {
                if ( (X % Y) != (X - (X/Y)*Y))
                    ok = false;
            }
        }
    }

    return ok;
}

//----------------------------------------------------------------

safeint32::Type operator %(const safeint32::Type& X, const safeint32::Type& Y)
{
    safeint32::Base Xv = INT32C__ACCESS(X);
    safeint32::Base Yv = INT32C__ACCESS(Y);

    safeint32::Base result = INT32C__INDEFINITE;

    if (Xv != INT32C__INDEFINITE && Yv != INT32C__INDEFINITE)
    {
        if (Yv != 0)
        {
            // dividing on any number (except zero) results in a valid range value
            COMPILE_ASSERT(safeint32::rangeIsSymmetric);

            // the remainder computation should work as tested by "builtinRemTest"
            result = Xv % Yv; 
        }
    }

    return INT32C__CREATE(result);
}
