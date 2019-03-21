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
    static inline void set(Uint& storage, uint32 value)
        {storage = (storage & ~mask) | (value << pos);}

    static inline bool equal(uint32 storage, uint32 value)
        {return (storage & mask) == (value << pos);}

    static inline uint32 get(uint32 storage)
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

    using Align = FormatBitfield<0, 2>;
    using Fill = FormatBitfield<2, 3>;
    using Base = FormatBitfield<3, 4>;
    using Plus = FormatBitfield<4, 5>;
    using Fform = FormatBitfield<5, 7>;

    using Width = FormatBitfield<8, 16>;
    using Precision = FormatBitfield<16, 24>;

public:

    inline FormatNumberOptions()
        {reset();}

    inline void reset()
    {
        uint32 v = 0;
        Precision::set(v, 6);
        theData = v;
    }

    //
    // Width
    //

    inline FormatNumberOptions& width(int32 v) {Width::set(theData, v); return *this;}
    inline int32 getWidth() const {return Width::get(theData);}

    //
    // Precision
    //

    inline FormatNumberOptions& precision(int32 v) {Precision::set(theData, v); return *this;}
    inline int32 getPrecision() const {return Precision::get(theData);}

    //
    // Fill
    //

    inline FormatNumberOptions& fillSpace() {Fill::set(theData, 0); return *this;}
    inline bool fillIsSpace() const {return Fill::equal(theData, 0);}

    inline FormatNumberOptions& fillZero() {Fill::set(theData, 1); return *this;}
    inline bool fillIsZero() const {return Fill::equal(theData, 1);}

    inline FormatNumberOptions& fillWith(char fill) {return fill == ' ' ? fillSpace() : fillZero();}
    inline char filledWith() const {return fillIsSpace() ? ' ' : '0';}

    //
    // Alignment
    //

    inline FormatNumberOptions& alignLeft() {Align::set(theData, 0); return *this;}
    inline bool alignIsLeft() const {return Align::equal(theData, 0);}

    inline FormatNumberOptions& alignRight() {Align::set(theData, 1); return *this;}
    inline bool alignIsRight() const {return Align::equal(theData, 1);}

    inline FormatNumberOptions& alignInternal() {Align::set(theData, 2); return *this;}
    inline bool alignIsInternal() const {return Align::equal(theData, 2);}

    //
    // Numbers base
    //

    inline FormatNumberOptions& baseDec() {Base::set(theData, 0); return *this;}
    inline bool baseIsDec() const {return Base::equal(theData, 0);}

    inline FormatNumberOptions& baseHex() {Base::set(theData, 1); return *this;}
    inline bool baseIsHex() const {return Base::equal(theData, 1);}

    //
    // Plus in front of non-negative numbers?
    //

    inline FormatNumberOptions& plusOff() {Plus::set(theData, 0); return *this;}
    inline bool plusIsOff() const {return Plus::equal(theData, 0);}

    inline FormatNumberOptions& plusOn() {Plus::set(theData, 1); return *this;}
    inline bool plusIsOn() const {return Plus::equal(theData, 1);}

    //
    // Float representation
    //

    inline FormatNumberOptions& fformF() {Fform::set(theData, 0); return *this;}
    inline bool fformIsF() const {return Fform::equal(theData, 0);}

    inline FormatNumberOptions& fformG() {Fform::set(theData, 1); return *this;}
    inline bool fformIsG() const {return Fform::equal(theData, 1);}

    inline FormatNumberOptions& fformE() {Fform::set(theData, 2); return *this;}
    inline bool fformIsE() const {return Fform::equal(theData, 2);}

};
