#pragma once

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
// SStringBuffer
//
//================================================================

template <typename Type>
class SStringBuffer
{

public:

    bool realloc(size_t newSize)
        {return true;}

    size_t size() const 
        {return currentSize;}

private:

    Type* currentPtr = nullptr;
    size_t currentSize = 0;

};

//================================================================
//
// SimpleStringEx
//
// A string with built-in invalid state, like NAN for float numbers.
//
//----------------------------------------------------------------
//
// If the string is invalid:
//
// * valid() returns false,
// * cstr() returns empty string,
// * size() returns 0.
//
//----------------------------------------------------------------
//
// On any operation, if there is not enough memory, the result is invalid.
//
// Any operation involving invalid inputs results in invalid output.
//
// Assignment operation, if fully successful, makes the string valid again.
//
// Clear function makes the string empty and valid.
//
//----------------------------------------------------------------
//
// Only valid strings are considered to be equal.
// An invalid string is not equal to any other string.
// Two invalid strings are not equal.
//
//================================================================

template <typename Type>
class SimpleStringEx
{

    using String = SimpleStringEx<Type>;

    //----------------------------------------------------------------
    //
    // Construct / Destruct.
    //
    //----------------------------------------------------------------

public:

    sysinline SimpleStringEx()
        {}

    sysinline ~SimpleStringEx()
        {deallocate();}

    //----------------------------------------------------------------
    //
    // Construct from a value.
    //
    //----------------------------------------------------------------

public:

    explicit sysinline SimpleStringEx(const String& that)
        {assign(that);}

    explicit sysinline SimpleStringEx(const Type* cstr)
        {assign(cstr);}

    explicit sysinline SimpleStringEx(const Type* bufferPtr, size_t bufferSize)
        {assign(bufferPtr, bufferSize);}

    explicit sysinline SimpleStringEx(const CharArrayEx<Type>& that)
        {assign(that.ptr, that.size);}

    //----------------------------------------------------------------
    //
    // Assign.
    //
    //----------------------------------------------------------------

public:

    sysinline String& operator =(const String& that)
        {assign(that); return *this;}

    sysinline String& operator =(const Type* cstr)
        {assign(cstr); return *this;}

    sysinline String& operator =(const CharArrayEx<Type>& that)
        {assign(that.ptr, that.size); return *this;}

    //----------------------------------------------------------------
    //
    // Access.
    //
    //----------------------------------------------------------------

public:

    const Type* cstr() const;
    size_t size() const;

public:

    sysinline operator CharArrayEx<Type> () const 
        {return CharArrayEx<Type>(cstr(), size());}

    sysinline CharArrayEx<Type> charArray() const
        {return CharArrayEx<Type>(cstr(), size());}

public:

    sysinline bool valid() const
        {return theOk;}

    sysinline void invalidate()
        {theOk = false;}

    sysinline String& clear()
        {deallocate(); theOk = true; return *this;}

    //----------------------------------------------------------------
    //
    // Append.
    //
    //----------------------------------------------------------------

public:

    void append(const Type* thatPtr, size_t thatSize);

    sysinline void append(const CharArrayEx<Type>& that)
        {append(that.ptr, that.size);}

    sysinline void append(const Type* cstr)
        {append(charArrayFromPtr(cstr));}

    sysinline void append(const SimpleStringEx<Type>& that)
        {that.valid() ? append(that.charArray()) : invalidate();}

public:

    template <typename That>
    sysinline String& operator +=(const That& that)
        {append(that); return *this;}

    template <typename That>
    sysinline String& operator <<(const That& that)
        {append(that); return *this;}

    //----------------------------------------------------------------
    //
    // Assign.
    //
    //----------------------------------------------------------------

    void assign(const Type* bufferPtr, size_t bufferSize);

    sysinline void assign(const CharArrayEx<Type>& that)
        {assign(that.ptr, that.size);}

    sysinline void assign(const Type* cstr)
        {assign(charArrayFromPtr(cstr));}

    sysinline void assign(const String& that)
        {that.valid() ? assign(that.charArray()) : invalidate();}
    
    //----------------------------------------------------------------
    //
    // Compare.
    //
    //----------------------------------------------------------------

public:

    friend sysinline bool operator ==(const String& a, const String& b)
        {return a.valid() && b.valid() && strEqual(a.charArray(), b.charArray());}

    ////

    friend sysinline bool operator ==(const String& a, const Type* b)
        {return a.valid() && strEqual(a.charArray(), charArrayFromPtr(b));}

    friend sysinline bool operator ==(const Type* a, const String& b)
        {return b.valid() && strEqual(charArrayFromPtr(a), b.charArray());}

    ////

    friend sysinline bool operator ==(const String& a, const CharArrayEx<Type>& b)
        {return a.valid() && strEqual(a.charArray(), b);}

    friend sysinline bool operator ==(const CharArrayEx<Type>& a, const String& b)
        {return b.valid() && strEqual(a, b.charArray());}

    //----------------------------------------------------------------
    //
    // Exchange.
    //
    //----------------------------------------------------------------

public:

    friend sysinline void exchange(String& A, String& B)
    {
        exchange(A.theOk, B.theOk);
        exchange(A.theData, B.theData);
    }

    //----------------------------------------------------------------
    //
    // Private.
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
    {return str.valid();}

//================================================================
//
// SimpleString
//
//================================================================

using SimpleString = SimpleStringEx<CharType>;
