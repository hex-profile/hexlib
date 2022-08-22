#include "simpleString.h"

#include <stdlib.h>

#include "formatting/formatStream.h"
#include "storage/rememberCleanup.h"
#include "numbers/int/intType.h"

//================================================================
//
// EmptyString
//
//================================================================

template <typename Type>
Type EmptyString<Type>::zeroElement = 0;

//================================================================
//
// InvalidString
//
//================================================================

template <typename Type>
Type InvalidString<Type>::zeroElement = 0;

//================================================================
//
// SStringBuffer::clear
//
//================================================================

template <typename Type>
void SStringBuffer<Type>::clear()
{
    if (currentSize)
        free(currentPtr);

    currentPtr = EmptyString<Type>::cstr();
    currentSize = 0;
}

//================================================================
//
// SStringBuffer::realloc
//
//================================================================

template <typename Type>
bool SStringBuffer<Type>::realloc(size_t newSize)
{
    //----------------------------------------------------------------
    //
    // Invalidate.
    //
    //----------------------------------------------------------------

    if (currentSize)
        free(currentPtr);

    currentPtr = InvalidString<Type>::cstr();
    currentSize = 0;

    //----------------------------------------------------------------
    //
    // Allocate.
    //
    //----------------------------------------------------------------

    constexpr size_t maxSize = (TYPE_MAX(size_t) / sizeof(Type)) - 1;
    ensure(newSize <= maxSize);

    ////

    Type* newPtr = EmptyString<Type>::cstr();
        
    if (newSize)
        newPtr = (Type*) malloc((newSize + 1) * sizeof(Type));

    ensure(newPtr != 0);

    //----------------------------------------------------------------
    //
    // Record success.
    //
    //----------------------------------------------------------------

    currentPtr = newPtr;
    currentSize = newSize;

    return true;
}

//================================================================
//
// SimpleStringEx::assign
//
//================================================================

template <typename Type>
void SimpleStringEx<Type>::assign(const Type* thatPtr, size_t thatSize)
{
    REMEMBER_CLEANUP_EX(invalidateOnError, buffer.invalidate());

    ////

    buffer.clear();

    ////

    if (thatSize != 0)
    {
        ensurev(buffer.realloc(thatSize));

        Type* bufferPtr = buffer.cstr();

        if (thatSize)
        {
            memcpy(bufferPtr, thatPtr, thatSize * sizeof(Type));
            bufferPtr += thatSize;
        }

        *bufferPtr = 0;
    }

    ////

    invalidateOnError.cancel();
}

//================================================================
//
// SimpleStringEx<Type>::append
//
//================================================================

template <typename Type>
void SimpleStringEx<Type>::append(const Type* thatPtr, size_t thatSize)
{
    REMEMBER_CLEANUP_EX(invalidateOnError, buffer.invalidate());

    ////

    ensurev(buffer.valid());

    ////

    if (thatSize != 0)
    {
        ensurev(thatSize <= TYPE_MAX(size_t) - buffer.size()); // Subtraction is always valid.

        ////

        SStringBuffer<Type> newBuffer;
        ensurev(newBuffer.realloc(buffer.size() + thatSize));
        Type* newPtr = newBuffer.cstr();

        ////

        if (buffer.size())
        {
            memcpy(newPtr, buffer.cstr(), buffer.size() * sizeof(Type));
            newPtr += buffer.size();
        }

        ////

        memcpy(newPtr, thatPtr, thatSize * sizeof(Type));
        newPtr += thatSize;

        ////

        *newPtr = 0;

        ////

        exchange(newBuffer, buffer);
    }

    ////

    invalidateOnError.cancel();
}

//================================================================
//
// formatOutput
//
//================================================================

template <>
void formatOutput(const SimpleStringEx<CharType>& value, FormatOutputStream& outputStream)
{
    outputStream << value.charArray();
}

//================================================================
//
// Instantiations.
//
//================================================================

template class SStringBuffer<char>;
template class SStringBuffer<wchar_t>;

template class SimpleStringEx<char>;
template class SimpleStringEx<wchar_t>;
