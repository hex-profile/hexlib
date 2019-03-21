#include "simpleString.h"

#include <string>
#include <memory>

#include "formatting/formatStream.h"

//================================================================
//
//
//
//================================================================

template <>
void formatOutput(const SimpleString& value, FormatOutputStream& outputStream)
{
    outputStream.write(charArrayFromPtr(value.cstr()));
}

//================================================================
//
// StringData
//
//================================================================

struct StringData : public std::basic_string<CharType>
{
    using Base = std::basic_string<CharType>;

    inline StringData(const CharType* bufPtr, size_t bufSize)
        : Base(bufPtr, bufSize) {}

    inline StringData(size_t bufSize, CharType fillValue)
        : Base(bufSize, fillValue) {}
};

//================================================================
//
// SimpleString::deallocate
//
//================================================================

void SimpleString::deallocate()
{
    try
    {
        if (theData) delete theData;
    }
    catch (const std::exception&)
    {
        // just in case, but only crazy STL can throw it in destructor
    }

    theData = 0;
}

//================================================================
//
// SimpleString::assign(const SimpleString& that)
//
//================================================================

void SimpleString::assign(const SimpleString& that)
{
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
            theData = new (std::nothrow) StringData(*that.theData);
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
// SimpleString::assign(const CharType* bufPtr, size_t bufSize)
//
//================================================================

void SimpleString::assign(const CharType* bufPtr, size_t bufSize)
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
            theData = new (std::nothrow) StringData(bufPtr, bufSize);
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
// SimpleString::assign(const CharType* cstr)
//
//================================================================

void SimpleString::assign(const CharType* cstr)
{
    assign(cstr, strlen(cstr));
}

//================================================================
//
// SimpleString::assign(CharType fillValue, size_t bufSize);
//
//================================================================

void SimpleString::assign(CharType fillValue, size_t bufSize)
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
            theData = new (std::nothrow) StringData(bufSize, fillValue);

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
// SimpleString::cstr
//
//================================================================

const CharType* SimpleString::cstr() const
{
    const CharType* result = CT("");

    if (theOk && theData)
    {
        try {result = theData->c_str();}
        catch (const std::exception&) {}
    }

    return result;
}

//================================================================
//
// SimpleString::length
//
//================================================================

size_t SimpleString::length() const
{
    size_t result = 0;

    ////

    if (theOk && theData)
    {
        try {result = theData->length();}
        catch (const std::exception&) {}
    }

    ////

    return result;
}

//================================================================
//
// SimpleString::operator +=
//
//================================================================

SimpleString& SimpleString::operator +=(const SimpleString& that)
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
            theData = new (std::nothrow) StringData(*that.theData);
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
// SimpleString::operator ==
//
//================================================================

bool operator ==(const SimpleString& A, const SimpleString& B)
{
    if_not (A.theOk && B.theOk)
        return false;

    bool AEmpty = (A.theData == 0 || A.theData->length() == 0);
    bool BEmpty = (B.theData == 0 || B.theData->length() == 0);

    if (AEmpty && BEmpty)
        return true;

    if_not (A.theData != 0 && B.theData != 0)
        return false;

    return (*A.theData) == (*B.theData);
}
