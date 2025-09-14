#include "keyBuffer.h"

#include "errorLog/convertExceptions.h"
#include "errorLog/errorLog.h"
#include "podVector/podVector.h"

//================================================================
//
// KeyBufferImpl
//
//================================================================

struct KeyBufferImpl : public KeyBuffer
{

    //----------------------------------------------------------------
    //
    // reserve
    //
    //----------------------------------------------------------------

    virtual void reserve(Space capacity, stdPars(Kit))
    {
        stdExceptBegin;

        REQUIRE(capacity >= 0);
        buffer.reserve(size_t(capacity));

        stdExceptEnd;
    }

    //----------------------------------------------------------------
    //
    // dealloc
    //
    //----------------------------------------------------------------

    virtual void dealloc()
    {
        decltype(buffer) tmp; // noexcept
        buffer.swap(tmp); // noexcept
    }

    //----------------------------------------------------------------
    //
    // receiveKey
    //
    //----------------------------------------------------------------

    virtual void receiveKey(const KeyEvent& key, stdPars(Kit))
    {
        stdExceptBegin;

        buffer.push_back(key);

        stdExceptEnd;
    }

    //----------------------------------------------------------------
    //
    // getBuffer
    //
    //----------------------------------------------------------------

    virtual Array<const KeyEvent> getBuffer() const
    {
        return makeArray(buffer.data(), Space(buffer.size()));
    }

    //----------------------------------------------------------------
    //
    // clearBuffer
    //
    //----------------------------------------------------------------

    virtual void clearBuffer()
    {
        buffer.clear(); // noexcept
    }

    //----------------------------------------------------------------
    //
    // Data.
    //
    //----------------------------------------------------------------

    PodVector<KeyEvent> buffer;

};

UniquePtr<KeyBuffer> KeyBuffer::create() {return makeUnique<KeyBufferImpl>();}
