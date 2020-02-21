#pragma once

#include <string.h>

#include "charType/charArray.h"
#include "charType/charType.h"
#include "compileTools/compileTools.h"
#include "numbers/int/intType.h"
#include "charType/strUtils.h"

//================================================================
//
// StringData
//
//================================================================

template <typename Type>
struct StringData;

//================================================================
//
// SimpleStringEx
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
// The error state is inherited when assigning another SimpleStringEx which is in the error state.
//
// The error state can be cleared in assignment, if operation is successful and assigned value is good.
//
// clear() resets the error state and makes the string empty.
//
// Assigning an empty string is guaranteed to reset error state.
//
//================================================================

template <typename Type>
class SimpleStringEx
{

public:

    sysinline SimpleStringEx()
        {}

    sysinline ~SimpleStringEx()
        {deallocate();}

public:

    sysinline SimpleStringEx(const SimpleStringEx<Type>& that)
        {assign(that);}

    sysinline SimpleStringEx(const Type* cstr)
        {assign(cstr);}

    sysinline SimpleStringEx(const Type* bufPtr, size_t bufLen)
        {assign(bufPtr, bufLen);}

    sysinline SimpleStringEx(const CharArrayEx<Type>& that)
        {assign(that.ptr, that.size);}

    sysinline SimpleStringEx(Type fillValue, size_t bufSize)
        {assign(fillValue, bufSize);}

public:

    sysinline auto& operator =(const SimpleStringEx<Type>& that)
        {assign(that); return *this;}

    sysinline auto& operator =(const Type* cstr)
        {assign(cstr); return *this;}

    sysinline auto& operator =(const CharArrayEx<Type>& that)
        {assign(that.ptr, that.size); return *this;}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline bool isOk() const
        {return theOk;}

    size_t length() const;

    sysinline size_t size() const
        {return length();}

public:

    const Type* cstr() const;

    sysinline operator const Type* () const 
        {return cstr();}

public:

    sysinline operator CharArrayEx<Type> () const 
        {return CharArrayEx<Type>(cstr(), length());}

    sysinline CharArrayEx<Type> charArray() const
        {return CharArrayEx<Type>(cstr(), length());}

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

    SimpleStringEx& operator +=(const SimpleStringEx<Type>& that);

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
    // Assign.
    //
    //----------------------------------------------------------------

    void assign(const SimpleStringEx<Type>& that);
    
    void assign(const Type* cstr);

    void assign(const Type* bufPtr, size_t bufLen);

    void assign(const CharArrayEx<Type>& that)
        {assign(that.ptr, that.size);}

    void assign(Type fillValue, size_t bufSize);

    //----------------------------------------------------------------
    //
    // Comparisons.
    //
    // Only valid strings are considered equal.
    // NAN string is not equal to any string, even to another NAN string.
    //
    //----------------------------------------------------------------

public:

    template <typename AnyType>
    friend bool simpleStrEqual(const SimpleStringEx<AnyType>& A, const SimpleStringEx<AnyType>& B);

    sysinline friend bool operator ==(const SimpleStringEx<Type>& A, const SimpleStringEx<Type>& B)
        {return simpleStrEqual(A, B);}

    sysinline friend bool operator ==(const SimpleStringEx<Type>& A, const Type* B)
        {return A.isOk() && strEqual(A.cstr(), B);}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    friend sysinline void exchange(SimpleStringEx<Type>& A, SimpleStringEx<Type>& B)
    {
        exchange(A.theOk, B.theOk);
        exchange(A.theData, B.theData);
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

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
    StringData<Type>* theData = nullptr;

    // Error absense flag;
    bool theOk = true;

};

//================================================================
//
// def
//
//================================================================

template <typename Type>
sysinline bool def(const SimpleStringEx<Type>& str)
    {return str.isOk();}

//================================================================
//
// operator +
//
//================================================================

template <typename Type>
sysinline SimpleStringEx<Type> operator +(const SimpleStringEx<Type>& X, const SimpleStringEx<Type>& Y)
{
    SimpleStringEx<Type> result(X); 
    result += Y; 
    return result;
}

template <typename Type, typename Other>
sysinline SimpleStringEx<Type> operator +(const SimpleStringEx<Type>& X, const Other& Y)
{
    SimpleStringEx<Type> result(X); 
    result += SimpleStringEx<Type>(Y); 
    return result;
}

template <typename Type, typename Other>
sysinline SimpleStringEx<Type> operator +(const Other& X, const SimpleStringEx<Type>& Y)
{
    SimpleStringEx<Type> result(X); 
    result += SimpleStringEx<Type>(Y); 
    return result;
}

//================================================================
//
// SimpleString
//
//================================================================

using SimpleString = SimpleStringEx<CharType>;
