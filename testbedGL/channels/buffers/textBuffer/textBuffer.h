#pragma once

#include "charType/charArray.h"
#include "storage/smartPtr.h"
#include "storage/adapters/callable.h"

namespace textBuffer {

//================================================================
//
// Char
//
//================================================================

using Char = CharType;

//================================================================
//
// TextBuffer
//
//================================================================

struct TextBuffer
{
    //----------------------------------------------------------------
    //
    // Init/deinit.
    //
    //----------------------------------------------------------------

    static UniquePtr<TextBuffer> create();
    virtual ~TextBuffer() {}

    virtual void clearMemory() =0;

    //----------------------------------------------------------------
    //
    // Data API.
    //
    //----------------------------------------------------------------

    virtual void clear() =0;

    virtual void addLine(const Char* ptr, size_t size) may_throw =0;

    virtual CharArrayEx<Char> getDataRef() =0;

    //----------------------------------------------------------------
    //
    // Buffer API.
    //
    //----------------------------------------------------------------

    virtual bool hasUpdates() const =0;

    virtual void reset() =0;

    virtual bool absorb(TextBuffer& other) =0;

    virtual void moveFrom(TextBuffer& other) =0;
};

//----------------------------------------------------------------

}

using textBuffer::TextBuffer;
