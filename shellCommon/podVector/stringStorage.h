#pragma once

#include "podVector/podVector.h"

//================================================================
//
// StringStorageEx
//
// String adapter based on PodVector.
//
//================================================================

template <typename Char>
class StringStorageEx : public PodVector<Char>
{

public:

    using Base = PodVector<Char>;
    using StringRef = CharArrayEx<Char>;

    using Base::Base;
    using Base::append;
    using Base::data;
    using Base::size;

public:

    sysinline operator StringRef() const
        {return {data(), size()};}

    sysinline StringRef str() const
        {return {data(), size()};}

    sysinline bool operator ==(const StringRef& that) const
        {return strEqual(str(), that);}

    sysinline void operator =(const StringRef& str) may_throw
        {Base::assign(str.ptr, str.size, false);} // No expected growth.

    sysinline void append(const StringRef& str) may_throw
        {Base::append(str.ptr, str.size, true);}

    sysinline auto& operator <<(const StringRef& str) may_throw
        {append(str); return *this;}

};
