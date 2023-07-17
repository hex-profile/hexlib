#pragma once

#include "numbers/int/intBase.h"

//================================================================
//
// FormatNumberOptions
//
// Highly-optimized format options (4 bytes),
// for tuning the output of built-in numeric types.
//
//================================================================

//================================================================
//
// FormatBitfield
//
//================================================================

template <int32 beg, int32 end>
struct FormatBitfield
{
    static const int32 pos = beg;
    static const int32 width = end - beg;
    static const uint32 mask = ((uint32(1) << width) - 1) << pos;

    template <typename Uint>
    static sysinline void set(Uint& storage, uint32 value)
        {storage = (storage & ~mask) | (value << pos);}

    static sysinline bool equal(uint32 storage, uint32 value)
        {return (storage & mask) == (value << pos);}

    static sysinline uint32 get(uint32 storage)
        {return (storage & mask) >> pos;}
};

//================================================================
//
// FormatNumberOptions
//
//================================================================

class FormatNumberOptions
{

private:

    uint32 theData;

private:

    using Fill = FormatBitfield<2, 3>;
    using Base = FormatBitfield<3, 4>;
    using Plus = FormatBitfield<4, 5>;
    using Fform = FormatBitfield<5, 7>;

    using Width = FormatBitfield<8, 16>;
    using Precision = FormatBitfield<16, 24>;

public:

    sysinline FormatNumberOptions()
        {reset();}

    sysinline void reset()
    {
        uint32 v = 0;
        Precision::set(v, 6);
        theData = v;
    }

    //
    // Width
    //

    sysinline FormatNumberOptions& width(int32 v) {Width::set(theData, v); return *this;}
    sysinline int32 getWidth() const {return Width::get(theData);}

    //
    // Precision
    //

    sysinline FormatNumberOptions& precision(int32 v) {Precision::set(theData, v); return *this;}
    sysinline int32 getPrecision() const {return Precision::get(theData);}

    //
    // Fill
    //

    sysinline FormatNumberOptions& fillSpace() {Fill::set(theData, 0); return *this;}
    sysinline bool fillIsSpace() const {return Fill::equal(theData, 0);}

    sysinline FormatNumberOptions& fillZero() {Fill::set(theData, 1); return *this;}
    sysinline bool fillIsZero() const {return Fill::equal(theData, 1);}

    sysinline FormatNumberOptions& fillWith(char fill) {return fill == ' ' ? fillSpace() : fillZero();}
    sysinline char filledWith() const {return fillIsSpace() ? ' ' : '0';}

    //
    // Numbers base
    //

    sysinline FormatNumberOptions& baseDec() {Base::set(theData, 0); return *this;}
    sysinline bool baseIsDec() const {return Base::equal(theData, 0);}

    sysinline FormatNumberOptions& baseHex() {Base::set(theData, 1); return *this;}
    sysinline bool baseIsHex() const {return Base::equal(theData, 1);}

    //
    // Plus in front of non-negative numbers?
    //

    sysinline FormatNumberOptions& plusOff() {Plus::set(theData, 0); return *this;}
    sysinline bool plusIsOff() const {return Plus::equal(theData, 0);}

    sysinline FormatNumberOptions& plusOn() {Plus::set(theData, 1); return *this;}
    sysinline bool plusIsOn() const {return Plus::equal(theData, 1);}

    //
    // Float representation
    //

    sysinline FormatNumberOptions& fformG() {Fform::set(theData, 0); return *this;}
    sysinline bool fformIsG() const {return Fform::equal(theData, 0);}

    sysinline FormatNumberOptions& fformF() {Fform::set(theData, 1); return *this;}
    sysinline bool fformIsF() const {return Fform::equal(theData, 1);}

    sysinline FormatNumberOptions& fformE() {Fform::set(theData, 2); return *this;}
    sysinline bool fformIsE() const {return Fform::equal(theData, 2);}

};
