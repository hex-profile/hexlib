#include "textBuffer.h"

#include "compileTools/blockExceptionsSilent.h"
#include "podVector/podVector.h"
#include "storage/rememberCleanup.h"

namespace textBuffer {

using namespace std;

//================================================================
//
// TextBufferImpl
//
//================================================================

struct TextBufferImpl : public TextBuffer
{

    virtual void clearMemory()
    {
        buffer.dealloc();
    }

    //----------------------------------------------------------------
    //
    // Buffer API.
    //
    //----------------------------------------------------------------

    virtual bool hasUpdates() const
    {
        return buffer.size() != 0;
    }

    virtual void reset()
    {
        buffer.clear();
    }

    virtual bool absorb(TextBuffer& other);

    virtual void moveFrom(TextBuffer& other)
    {
        auto& that = (TextBufferImpl&) other;

        buffer.swap(that.buffer);
        that.buffer.clear();
    }

    //----------------------------------------------------------------
    //
    // Data API.
    //
    //----------------------------------------------------------------

    virtual void clear()
    {
        buffer.clear();
    }

    virtual void addLine(const Char* ptr, size_t size)
    {
        buffer.append(ptr, size, true);
        buffer.push_back('\n');
    }

    virtual CharArrayEx<Char> getDataRef()
    {
        return {buffer.data(), buffer.size()};
    }

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    PodVector<Char> buffer;

};

UniquePtr<TextBuffer> TextBuffer::create() {return makeUnique<TextBufferImpl>();}

//================================================================
//
// TextBufferImpl::absorb
//
//================================================================

bool TextBufferImpl::absorb(TextBuffer& other)
{
    boolFuncExceptBegin;

    auto& that = (TextBufferImpl&) other;

    REMEMBER_CLEANUP(that.reset());

    ////

    if (buffer.size() == 0)
    {
        moveFrom(other); // Full content replacement.
        return true;
    }

    ////

    auto& src = that.buffer;

    buffer.append(src.data(), src.size(), false);

    ////

    boolFuncExceptEnd;
}

//----------------------------------------------------------------

}
