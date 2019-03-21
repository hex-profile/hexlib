#include "numbers/safeint32/safeint32.h"

namespace safeint32CodeTest {

//================================================================
//
// Code Test
//
//================================================================

safeint32::Type TestPlus(safeint32::Type X)
    {return +X;}

safeint32::Type TestMinus(safeint32::Type X)
    {return -X;}

safeint32::Type TestAdd(safeint32::Type X, safeint32::Type Y)
    {return X + Y;}

safeint32::Type TestSub(safeint32::Type X, safeint32::Type Y)
    {return X - Y;}

safeint32::Type TestMul(safeint32::Type X, safeint32::Type Y)
    {return X * Y;}

safeint32::Type TestDiv(safeint32::Type X, safeint32::Type Y)
    {return X / Y;}

safeint32::Type TestRem(safeint32::Type X, safeint32::Type Y)
    {return X % Y;}

safeint32::Type TestShl(safeint32::Type X, safeint32::Type Y)
    {return X << Y;}

safeint32::Type TestShr(safeint32::Type X, safeint32::Type Y)
    {return X >> Y;}

safeint32::Type TestBitnot(safeint32::Type X)
    {return ~X;}

safeint32::Type TestBitand(safeint32::Type X, safeint32::Type Y)
    {return X & Y;}

safeint32::Type TestBitor(safeint32::Type X, safeint32::Type Y)
    {return X | Y;}

//================================================================
//
// 
//
//================================================================

safeint32::Type TestInline(safeint32::Type AX, safeint32::Type AY, safeint32::Type BX, safeint32::Type BY)
{
    // 50
    return (AX*BX + AY*BY) * (AX + BX) * (AY + BY);
}

//----------------------------------------------------------------

safeint32::Type TestNoinline(safeint32::Type AX, safeint32::Type AY, safeint32::Type BX, safeint32::Type BY)
{
    // 113 (33)
    return TestMul(TestMul(TestAdd(TestMul(AX, BX), TestMul(AY, BY)), TestAdd(AX, BX)), TestAdd(AY, BY));
}

//----------------------------------------------------------------

}
