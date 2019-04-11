#pragma once

#include <string.h>

#include "charType/charType.h"
#include "compileTools/compileTools.h"
#include "numbers/interface/numberInterface.h"
#include "charType/charArray.h"
#include "numbers/int/intType.h"

//================================================================
//
// SimpleString
//
// String with built-in error state.
//
//----------------------------------------------------------------
//
// If the string is in error state:
//
// cstr() returns empty string.
// operator bool() returns false.
// length() returns 0.
//
// On any operation, if you do not have enough memory, the string sets error state.
// Also error state is inherited when assigning another SimpleString, which is in error state.
//
// Error state can be cleared in assignment, if operation is successful and assigned value
//
// clear() resets the error state assuredly and makes the string empty.
//
// Assigning an empty string also is guaranteed to reset error state.
//
//================================================================

class SimpleString
{

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

private:

    // Content ptr
    struct StringData* theData;

    // Error absense flag;
    bool theOk;

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

private:

    inline void initializeEmpty()
        {theData = 0; theOk = true;}

    void deallocate();

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    inline SimpleString()
        {initializeEmpty();}

    inline ~SimpleString()
        {deallocate();}

public:

    inline SimpleString(const SimpleString& that)
        {initializeEmpty(); assign(that);}

    inline SimpleString& operator =(const SimpleString& that)
        {assign(that); return *this;}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    inline SimpleString(const CharType* cstr)
        {initializeEmpty(); assign(cstr);}

    inline SimpleString(const CharType* bufPtr, size_t bufLen)
        {initializeEmpty(); assign(bufPtr, bufLen);}

    inline SimpleString(const CharArray& str)
        {initializeEmpty(); assign(str.ptr, str.size);}

    inline SimpleString& operator =(const CharType* cstr)
        {assign(cstr); return *this;}

    inline SimpleString& operator =(const CharArray& str)
        {assign(str.ptr, str.size); return *this;}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    inline bool ok() const
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

    inline void invalidate()
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

    inline void clear()
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

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    friend bool operator ==(const SimpleString& A, const SimpleString& B);

    inline friend bool operator !=(const SimpleString& A, const SimpleString& B)
        {return !(A == B);}

public:

    friend inline void exchange(SimpleString& A, SimpleString& B)
    {
        exchange(A.theOk, B.theOk);
        exchange(A.theData, B.theData);
    }

};

//================================================================
//
//
//
//================================================================

inline SimpleString operator +(const SimpleString& X, const SimpleString& Y)
    {SimpleString result(X); result += Y; return result;}

inline SimpleString operator +(const SimpleString& X, const CharType* Y)
    {SimpleString result(X); result += SimpleString(Y); return result;}

inline SimpleString operator +(const CharType* X, const SimpleString& Y)
    {SimpleString result(X); result += Y; return result;}
