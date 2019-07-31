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
    outputStream.write(charArrayFromPtr(value.cstr()));
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

    sysinline StringData(const Type* bufPtr, size_t bufSize)
        : Base(bufPtr, bufSize) {}

    sysinline StringData(size_t bufSize, Type fillValue)
        : Base(bufSize, fillValue) {}
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
// SimpleStringEx<Type>::assign(const SimpleStringEx<Type>& that)
//
//================================================================

template <typename Type>
void SimpleStringEx<Type>::assign(const SimpleStringEx<Type>& that)
{
    if (this == &that)
        return;

    ////

    theOk = false;

    ////

    if_not (that.theOk)
        return;

    ////

    if_not (that.theData) // empty string
    {
        deallocate();
        theOk = true;
        return;
    }

    ////

    try
    {
        if (theData != 0)
            *theData = *that.theData;
        else
        {
            theData = new (std::nothrow) StringData<Type>(*that.theData);
            if_not (theData)
                return;
        }
    }
    catch (const std::exception&) {}

    ////

    this->theOk = true;
}

//================================================================
//
// SimpleStringEx<Type>::assign(const Type* bufPtr, size_t bufSize)
//
//================================================================

template <typename Type>
void SimpleStringEx<Type>::assign(const Type* bufPtr, size_t bufSize)
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
            (*theData).assign(bufPtr, bufSize);
        else
        {
            theData = new (std::nothrow) StringData<Type>(bufPtr, bufSize);

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
// SimpleStringEx<Type>::assign(const Type* cstr)
//
//================================================================

void SimpleStringEx<char>::assign(const char* cstr)
    {assign(cstr, strlen(cstr));}

void SimpleStringEx<wchar_t>::assign(const wchar_t* cstr)
    {assign(cstr, wcslen(cstr));}

//================================================================
//
// SimpleStringEx<Type>::assign(Type fillValue, size_t bufSize);
//
//================================================================

template <typename Type>
void SimpleStringEx<Type>::assign(Type fillValue, size_t bufSize)
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
            (*theData).assign(bufSize, fillValue);
        else
        {
            theData = new (std::nothrow) StringData<Type>(bufSize, fillValue);

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
// SimpleStringEx<Type>::length
//
//================================================================

template <typename Type>
size_t SimpleStringEx<Type>::length() const
{
    size_t result = 0;

    if (theOk && theData)
        result = theData->length();

    return result;
}

//================================================================
//
// SimpleStringEx<Type>::operator +=
//
//================================================================

template <typename Type>
SimpleStringEx<Type>& SimpleStringEx<Type>::operator +=(const SimpleStringEx<Type>& that)
{
    if_not (this->theOk && that.theOk)
    {
        theOk = false;
        return *this;
    }

    ////

    if_not (that.theData)
        return *this; // everything is done

    ////

    try
    {
        if (theData != 0)
        {
            (*theData) += (*that.theData);
        }
        else
        {
            theData = new (std::nothrow) StringData<Type>(*that.theData);
            if_not (theData) {theOk = false; return *this;}
        }
    }
    catch (const std::exception&)
    {
        theOk = false;
    }

    ////

    return *this;
}

//================================================================
//
// SimpleStringEx<Type>::operator ==
//
//================================================================

template <typename Type>
bool stringsEqual(const SimpleStringEx<Type>& A, const SimpleStringEx<Type>& B)
{
    if_not (def(A) && def(B))
        return false;

    bool filledA = (A.length() != 0);
    bool filledB = (B.length() != 0);

    if (!filledA && !filledB)
        return true;

    if_not (filledA && filledB)
        return false;

    return (*A.theData) == (*B.theData);
}

//================================================================
//
// Instantiations.
//
//================================================================

template class SimpleStringEx<char>;
template class SimpleStringEx<wchar_t>;

INSTANTIATE_FUNC(stringsEqual<char>)
INSTANTIATE_FUNC(stringsEqual<wchar_t>)
