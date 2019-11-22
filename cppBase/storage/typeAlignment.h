#pragma once

//================================================================
//
// MaxNaturalAlignment
//
// The struct forcing maximum alignment of built-in types.
//
//================================================================

union MaxNaturalAlignment
{
    void* alignPtr;
    
    char alignChar;
    signed char alignSignedChar;
    unsigned char alignUnsignedChar;

    signed short alignSignedShort;
    unsigned short alignUnsignedShort;

    signed int alignSignedInt;
    unsigned int alignUnsignedInt;

    signed long alignSignedLong;
    unsigned long alignUnsignedLong;

    float alignFloat;
    double alignDouble;
};

//----------------------------------------------------------------

constexpr size_t maxNaturalAlignment = alignof(MaxNaturalAlignment);
