#pragma once

#include "charType/charArray.h"
#include "storage/noncopyable.h"
#include "numbers/interface/exchangeInterface.h"

//================================================================
//
// EmptyString
//
//================================================================

template <typename Type>
struct EmptyString
{

public:

    static sysinline Type* cstr()
        {return &zeroElement;}

private:

    // Writable, but only zero value may be written.
    static Type zeroElement;

};

//================================================================
//
// InvalidString
//
//================================================================

template <typename Type>
struct InvalidString
{

public:

    static sysinline Type* cstr()
        {return &zeroElement;}

private:

    // Writable, but only zero value may be written.
    static Type zeroElement;

};

//================================================================
//
// SStringBuffer
//
// String buffer with trailing NUL character.
//
//================================================================

template <typename Type>
class SStringBuffer : public NonCopyable
{
    
public:
    
    sysinline ~SStringBuffer()
        {clear();}

public:

    sysinline size_t size() const 
        {return currentSize;}

    sysinline Type* cstr() const
        {return currentPtr;}

public:

    void clear();

    bool realloc(size_t newSize);

public:

    sysinline bool valid() const 
    {
        return currentPtr != InvalidString<Type>::cstr();
    }

    sysinline void invalidate()
    {
        clear();

        currentPtr = InvalidString<Type>::cstr();
    }

public:

    friend sysinline void exchange(SStringBuffer<Type>& a, SStringBuffer<Type>& b)
    {
        exchangeByCopying(a.currentPtr, b.currentPtr);
        exchangeByCopying(a.currentSize, b.currentSize);
    }

private:

    //
    // If the size is zero, the pointer is NOT allocated.
    //

    Type* currentPtr = EmptyString<Type>::cstr();
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
    // Construct.
    //
    //----------------------------------------------------------------

public:

    sysinline SimpleStringEx()
        {}

    //----------------------------------------------------------------
    //
    // Construct from a value.
    //
    //----------------------------------------------------------------

public:

    sysinline SimpleStringEx(const String& that)
        {assign(that);}

    explicit sysinline SimpleStringEx(const Type* cstr)
        {assign(cstr);}

    explicit sysinline SimpleStringEx(const Type* bufferPtr, size_t bufferSize)
        {assign(bufferPtr, bufferSize);}

    explicit sysinline SimpleStringEx(const CharArrayEx<Type>& that)
        {assign(that);}

    //----------------------------------------------------------------
    //
    // Access.
    //
    //----------------------------------------------------------------

public:

    sysinline const Type* cstr() const
        {return buffer.cstr();}

    sysinline size_t size() const
        {return buffer.size();}

public:

    sysinline operator CharArrayEx<Type> () const 
        {return CharArrayEx<Type>(cstr(), size());}

    sysinline CharArrayEx<Type> charArray() const
        {return CharArrayEx<Type>(cstr(), size());}

public:

    sysinline bool valid() const
        {return buffer.valid();}

    sysinline void invalidate()
        {buffer.invalidate();}

    sysinline String& clear()
        {buffer.clear(); return *this;}

    //----------------------------------------------------------------
    //
    // Assign.
    //
    //----------------------------------------------------------------

public:

    void assign(const Type* bufferPtr, size_t bufferSize);

    sysinline void assign(const CharArrayEx<Type>& that)
        {assign(that.ptr, that.size);}

    sysinline void assign(const Type* cstr)
        {assign(charArrayFromPtr(cstr));}

    sysinline void assign(const String& that)
        {that.valid() ? assign(that.charArray()) : invalidate();}

public:

    sysinline String& operator =(const String& that)
        {assign(that); return *this;}

    sysinline String& operator =(const Type* cstr)
        {assign(cstr); return *this;}

    sysinline String& operator =(const CharArrayEx<Type>& that)
        {assign(that); return *this;}

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
    sysinline String& operator <<(const That& that)
        {append(that); return *this;}

    template <typename That>
    sysinline String& operator +=(const That& that)
        {append(that); return *this;}

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

    friend sysinline void exchange(String& a, String& b)
    {
        exchange(a.buffer, b.buffer);
    }

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    SStringBuffer<Type> buffer;

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
