#pragma once

#include <string.h>

#include "charType/charArray.h"
#include "charType/charType.h"
#include "compileTools/compileTools.h"
#include "numbers/int/intType.h"

//================================================================
//
// SimpleString
//
// String with built-in error state (like NAN for numbers).
//
//----------------------------------------------------------------
//
// If the string is in the error state:
//
// * cstr() returns empty string,
// * isOk() returns false,
// * length() returns 0.
//
// At any operation, if there is not enough memory, a string sets the error state.
// The error state is inherited when assigning another SimpleString which is in the error state.
//
// The error state can be cleared in assignment, if operation is successful and assigned value is good.
//
// clear() resets the error state and makes the string empty.
//
// Assigning an empty string is guaranteed to reset error state.
//
//================================================================

class SimpleString
{

public:

    sysinline SimpleString()
        {}

    sysinline ~SimpleString()
        {deallocate();}

public:

    sysinline SimpleString(const SimpleString& that)
        {assign(that);}

    sysinline SimpleString& operator =(const SimpleString& that)
        {assign(that); return *this;}

public:

    template <typename That>
    sysinline SimpleString(const That& that)
        {assign(that);}

    template <typename That1, typename That2>
    sysinline SimpleString(const That1& that1, const That2& that2)
        {assign(that1, that2);}

    template <typename That>
    sysinline SimpleString& operator =(const That& that)
        {assign(that); return *this;}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline bool isOk() const
        {return theOk;}

    size_t length() const;

    const CharType* cstr() const;

    CharArray charArray() const
        {return CharArray(cstr(), length());}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline void invalidate()
        {theOk = false;}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    SimpleString& operator +=(const SimpleString& that);

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline void clear()
        {deallocate(); theOk = true;}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    void assign(const SimpleString& that);
    void assign(const CharType* cstr);
    void assign(const CharType* bufPtr, size_t bufLen);
    void assign(CharType fillValue, size_t bufSize);

    void assign(const CharArray& that)
        {assign(that.ptr, that.size);}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    friend bool operator ==(const SimpleString& A, const SimpleString& B);

    sysinline friend bool operator !=(const SimpleString& A, const SimpleString& B)
        {return !(A == B);}

public:

    friend sysinline void exchange(SimpleString& A, SimpleString& B)
    {
        exchange(A.theOk, B.theOk);
        exchange(A.theData, B.theData);
    }

private:

    void deallocate();

private:

    //
    // theOk == true && theData == 0:
    // Empty string is valid.
    //
    // theOk == true && theData != 0:
    // Valid string is contained in theData.
    //
    // theOk == false && any theData
    // String is invalid and blank
    //

    // Content ptr
    struct StringData* theData = nullptr;

    // Error absense flag;
    bool theOk = true;

};

//================================================================
//
// def
//
//================================================================

sysinline bool def(const SimpleString& str)
    {return str.isOk();}

//================================================================
//
// operator +
//
//================================================================

sysinline SimpleString operator +(const SimpleString& X, const SimpleString& Y)
{
    SimpleString result(X); 
    result += Y; 
    return result;
}

sysinline SimpleString operator +(const SimpleString& X, const CharType* Y)
    {SimpleString result(X); result += SimpleString(Y); return result;}

sysinline SimpleString operator +(const CharType* X, const SimpleString& Y)
    {SimpleString result(X); result += Y; return result;}
