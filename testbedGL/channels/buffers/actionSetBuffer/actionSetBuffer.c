#include "actionSetBuffer.h"

#include "compileTools/blockExceptionsSilent.h"
#include "channels/buffers/actionSetBuffer/greedyVector.h"
#include "podVector/stringStorage.h"

namespace actionSetBuffer {

using namespace std;

//================================================================
//
// ActionRecord
//
//================================================================

class ActionRecord
{

public:

    sysinline ActionRecord() =default;

    ////

    sysinline ActionRecord& operator=(const ActionRecord& that) =delete;
    sysinline ActionRecord(const ActionRecord& that) =delete;

    ////

    sysinline ActionRecord(ActionRecord&& that)
        {moveFrom(that);}

    sysinline ActionRecord& operator=(ActionRecord&& that)
        {moveFrom(that); return *this;}

    ////

    sysinline void moveFrom(ActionRecord& that)
    {
        id = that.id;
        key.swap(that.key);
        name.swap(that.name);
        comment.swap(that.comment);

        that.id = 0;
        that.key.clear();
        that.name.clear();
        that.comment.clear();
    }

    ////

    sysinline void setAll(ActionId id, CharArray name, CharArray key, CharArray comment)
    {
        bool fast = true;
        fast &= (key.size <= this->key.capacity());
        fast &= (name.size <= this->name.capacity());
        fast &= (comment.size <= this->comment.capacity());

        this->id = id;
        this->key = key;
        this->name = name;
        this->comment = comment;
    }

    sysinline ActionId getId() const
        {return id;}

    sysinline CharArray getKey() const
        {return key;}

    sysinline CharArray getName() const
        {return name;}

    sysinline CharArray getComment() const
        {return comment;}

private:

    ActionId id = 0;

    StringStorageEx<CharType> key;
    StringStorageEx<CharType> name;
    StringStorageEx<CharType> comment;

};

//================================================================
//
// ActionSetBufferImpl
//
//================================================================

struct ActionSetBufferImpl : public ActionSetBuffer
{

    virtual void clearMemory()
    {
        cleared = false;
        buffer.clearMemory();
    }

    //----------------------------------------------------------------
    //
    // Buffer API.
    //
    //----------------------------------------------------------------

    virtual bool hasUpdates() const
    {
        return
            cleared ||
            buffer.size() != 0;
    }

    void reset()
    {
        cleared = false;
        buffer.clear();
    }

    virtual bool absorb(ActionSetBuffer& other);

    virtual void moveFrom(ActionSetBuffer& other);

    //----------------------------------------------------------------
    //
    // Data API.
    //
    //----------------------------------------------------------------

    virtual void dataClear()
    {
        cleared = true;
        buffer.clear();
    }

    ////

    virtual bool dataAdd(ActionId id, CharArray name, CharArray key, CharArray comment)
    {
        boolFuncExceptBegin;

        buffer.appendAtEnd(1);
        REMEMBER_CLEANUP_EX(errorCleanup, buffer.removeFromEnd(1));

        auto& r = *(buffer.end() - 1);
        r.setAll(id, name, key, comment);

        errorCleanup.cancel();

        boolFuncExceptEnd;
    }

    ////

    virtual size_t dataCount() const
    {
        return buffer.size();
    }

    virtual void dataGet(ActionRecordReceiver& receiver) const
    {
        for (auto& r : buffer)
            receiver(r.getId(), r.getName(), r.getKey(), r.getComment());
    }

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    bool cleared = false;

    GreedyVector<ActionRecord> buffer;

};

UniquePtr<ActionSetBuffer> ActionSetBuffer::create() {return makeUnique<ActionSetBufferImpl>();}

//================================================================
//
// ActionSetBufferImpl::absorb
//
//================================================================

bool ActionSetBufferImpl::absorb(ActionSetBuffer& other)
{
    boolFuncExceptBegin;

    auto& that = (ActionSetBufferImpl&) other;

    REMEMBER_CLEANUP(that.reset());

    //
    // Absorb clearing.
    //

    if (that.cleared)
        dataClear();

    //
    // Absorb buffer.
    //

    if (that.buffer.size() == 0)
        ;
    else if (buffer.size() == 0)
        buffer.moveFrom(that.buffer); // Full content replacement.
    else
        appendSeq(buffer, that.buffer.begin(), that.buffer.size());

    ////

    boolFuncExceptEnd;
}

//================================================================
//
// ActionSetBufferImpl::moveFrom
//
//================================================================

void ActionSetBufferImpl::moveFrom(ActionSetBuffer& other)
{
    auto& that = (ActionSetBufferImpl&) other;

    REMEMBER_CLEANUP(that.reset());

    ////

    this->cleared = that.cleared;
    this->buffer.moveFrom(that.buffer); // no-throw
}

//----------------------------------------------------------------

}
