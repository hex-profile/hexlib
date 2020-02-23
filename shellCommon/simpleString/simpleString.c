#include "simpleString.h"

#include <string>
#include <memory>

#include "formatting/formatStream.h"

//================================================================
//
// formatOutput
//
//================================================================

template <>
void formatOutput(const SimpleStringEx<CharType>& value, FormatOutputStream& outputStream)
{
    outputStream.write(value.charArray());
}

//================================================================
//
// StringData
//
//================================================================

template <typename Type>
struct StringData : public std::basic_string<Type>
{
    using Base = std::basic_string<Type>;

    sysinline StringData(const Type* bufferPtr, size_t bufSize)
        : Base(bufferPtr, bufSize) {}
};

//================================================================
//
// SimpleStringEx<Type>::deallocate
//
//================================================================

template <typename Type>
void SimpleStringEx<Type>::deallocate()
{
    try
    {
        delete theData;
    }
    catch (const std::exception&)
    {
        // just in case, but only crazy STL can throw it in destructor
    }

    theData = nullptr;
}

//================================================================
//
// SimpleStringEx<Type>::assign(const Type* bufferPtr, size_t bufSize)
//
//================================================================

template <typename Type>
void SimpleStringEx<Type>::assign(const Type* bufferPtr, size_t bufSize)
{
    theOk = false;

    ////

    if (bufSize == 0) // empty string
    {
        deallocate();
        theOk = true;
        return;
    }

    ////

    try
    {
        if (theData != 0)
            (*theData).assign(bufferPtr, bufSize);
        else
        {
            theData = new (std::nothrow) StringData<Type>(bufferPtr, bufSize);

            if_not (theData)
                return;
        }
    }
    catch (const std::exception&) {}

    ////

    theOk = true;
}

//================================================================
//
// SimpleStringEx<Type>::cstr
//
//================================================================

template <typename Type>
const Type* SimpleStringEx<Type>::cstr() const
{
    static const Type emptyStr[] = {0};

    const Type* result = emptyStr;

    if (theOk && theData)
        result = theData->c_str();

    return result;
}

//================================================================
//
// SimpleStringEx<Type>::size
//
//================================================================

template <typename Type>
size_t SimpleStringEx<Type>::size() const
{
    size_t result = 0;

    if (theOk && theData)
        result = theData->size();

    return result;
}

//================================================================
//
// SimpleStringEx<Type>::append
//
//================================================================

template <typename Type>
void SimpleStringEx<Type>::append(const Type* thatPtr, size_t thatSize)
{
    if_not (this->valid())
        return; // Keep invalid.

    ////

    if (thatSize == 0)
        return; // Done.

    ////

    try
    {
        if (theData != 0)
        {
            (*theData).append(thatPtr, thatSize);
        }
        else
        {
            theData = new (std::nothrow) StringData<Type>(thatPtr, thatSize);
            if_not (theData) {theOk = false; return;}
        }
    }
    catch (const std::exception&)
    {
        theOk = false;
    }
}

//================================================================
//
// Instantiations.
//
//================================================================

template class SimpleStringEx<char>;
template class SimpleStringEx<wchar_t>;
