#include "logBuffer.h"

#include <deque>
#include <stdexcept>

#include "numbers/int/intType.h"
#include "storage/rememberCleanup.h"
#include "errorLog/debugBreak.h"
#include "podVector/stringStorage.h"

using namespace std;

//================================================================
//
// STRESS_TEST
//
//================================================================

#define STRESS_TEST 0

////

#if !STRESS_TEST
    #define STRESS_TEST_POINT(condition)
#else
    #define STRESS_TEST_POINT(condition) \
        if (condition) throw bad_alloc();
#endif

//================================================================
//
// THROW_REQUIRE
//
//================================================================

[[noreturn]]
void throwInternalError()
{
    throw runtime_error("Internal error");
}

//----------------------------------------------------------------

#define THROW_REQUIRE(condition) \
    if (condition) ; else throwInternalError();

//================================================================
//
// BasicBufferSimple
//
//================================================================

template <typename Type>
class BasicBufferSimple
{

public:

    sysinline BasicBufferSimple() =default;

    ////

    sysinline BasicBufferSimple(const BasicBufferSimple& that) =delete;
    sysinline BasicBufferSimple& operator=(const BasicBufferSimple& that) =delete;

    ////

    void clearMemory()
    {
        data.clear();
    }

    ////

    sysinline void moveFrom(BasicBufferSimple& that)
    {
        data.swap(that.data);
        that.clear();
    }

    ////

    sysinline auto size() const
        {return data.size();}

    sysinline void clear()
        {data.clear();}

    ////

    sysinline auto begin()
        {return data.begin();}

    sysinline auto end()
        {return data.end();}

    ////

    sysinline void cutToLimit(size_t limit) // no-throw
    {
        if (data.size() > limit)
        {
            auto extraItems = data.size() - limit;
            data.erase(data.begin(), data.begin() + extraItems);
        }
    }

    ////

    void removeFromBeginningAndAppendAtEnd(size_t removeCount, size_t appendCount);

    ////

    void removeFromEnd(size_t n) // no-throw
    {
        n = clampMax(n, data.size());
        data.erase(data.end() - n, data.end());
    }

private:

    deque<Type> data;

};

//================================================================
//
// BasicBufferSimple::removeFromBeginningAndAppendAtEnd
//
//================================================================

template <typename Type>
void BasicBufferSimple<Type>::removeFromBeginningAndAppendAtEnd(size_t removeCount, size_t appendCount)
{
    removeCount = clampMax(removeCount, data.size());

    //----------------------------------------------------------------
    //
    // Surprisingly deque::resize is not atomic, so the rollback.
    //
    //----------------------------------------------------------------

    auto originalSize = data.size();

    ////

    if (appendCount)
    {
        REMEMBER_CLEANUP_EX(appendRollback, data.resize(originalSize)); // no-throw

        ////

        STRESS_TEST_POINT(rand() % 13 == 0)
        data.resize(data.size() + appendCount);

        ////

        appendRollback.cancel();
    }

    //----------------------------------------------------------------
    //
    // Move shared items: Shouldn't throw.
    // This is made to preserve string allocated capacity.
    //
    //----------------------------------------------------------------

    auto sharedCount = minv(removeCount, appendCount);

    if (sharedCount)
    {
        auto srcPtr = data.begin();
        auto dstPtr = data.begin() + originalSize;

        for_count (i, sharedCount)
            *dstPtr++ = std::move(*srcPtr++); // Should be no-throw
    }

    //----------------------------------------------------------------
    //
    // Remove (no-throw).
    //
    //----------------------------------------------------------------

    if (removeCount)
        data.erase(data.begin(), data.begin() + removeCount);
}

//================================================================
//
// BasicBufferGreedy
//
// Preserves capacity like vector<>.
//
//================================================================

template <typename Type>
class BasicBufferGreedy
{

public:

    sysinline BasicBufferGreedy() =default;

    sysinline BasicBufferGreedy(const BasicBufferGreedy& that) =delete;
    sysinline BasicBufferGreedy& operator=(const BasicBufferGreedy& that) =delete;

    ////

    sysinline void clearMemory()
    {
        count = 0;
        data.clear();
    }

    ////

    sysinline void moveFrom(BasicBufferGreedy& that)
    {
        data.swap(that.data);
        count = that.count;
        that.count = 0;
    }

    ////

    sysinline auto size() const
        {return count;}

    sysinline void clear()
        {count = 0;}

    ////

    sysinline auto begin()
        {return data.begin();}

    sysinline auto end()
        {return data.begin() + count;}

    ////

    void cutToLimit(size_t limit);

    ////

    void removeFromBeginningAndAppendAtEnd(size_t removeCount, size_t appendCount);

    ////

    void removeFromEnd(size_t n) // no-throw
    {
        n = clampMax(n, count);
        count -= n;
    }

private:

    size_t count = 0; // Always <= size()

    deque<Type> data;

};

//================================================================
//
// BasicBufferGreedy::cutToLimit
//
//================================================================

template <typename Type>
void BasicBufferGreedy<Type>::cutToLimit(size_t limit)
{
    if (data.size() <= limit)
        return;

    auto removeCount = data.size() - limit;

    ////

    auto removeFromReserve = minv(data.size() - count, removeCount);

    if (removeFromReserve)
    {
        data.erase(data.end() - removeFromReserve, data.end()); // no-throw
        removeCount -= removeFromReserve;
    }

    ////

    if (removeCount)
    {
        DEBUG_BREAK_CHECK(count == data.size()); // Reserve is exhausted
        data.erase(data.begin(), data.begin() + removeCount); // no-throw
        count -= removeCount;
    }
}

//================================================================
//
// BasicBufferGreedy::removeFromBeginningAndAppendAtEnd
//
//================================================================

template <typename Type>
void BasicBufferGreedy<Type>::removeFromBeginningAndAppendAtEnd(size_t removeCount, size_t appendCount)
{
    removeCount = clampMax(removeCount, count);

    //----------------------------------------------------------------
    //
    // Append.
    //
    //----------------------------------------------------------------

    auto originalSize = data.size();
    auto originalCount = count;

    REMEMBER_CLEANUP_EX(appendRollback, {data.resize(originalSize); count = originalCount;});

    ////

    auto reserveAppendCount = minv(data.size() - count, appendCount);

    if (reserveAppendCount)
        count += reserveAppendCount;

    ////

    auto allocAppendCount = appendCount - reserveAppendCount;

    if (allocAppendCount)
    {
        STRESS_TEST_POINT(rand() % 13 == 0)
        data.resize(data.size() + allocAppendCount);
        count += allocAppendCount;
    }

    ////

    appendRollback.cancel();

    //----------------------------------------------------------------
    //
    // Move shared items: Shouldn't throw.
    // This is made to preserve string allocated capacity.
    //
    //----------------------------------------------------------------

    auto sharedCount = minv(removeCount, appendCount);

    if (sharedCount)
    {
        auto srcPtr = data.begin();
        auto dstPtr = data.begin() + originalSize;

        for_count (i, sharedCount)
            *dstPtr++ = move(*srcPtr++);
    }

    //----------------------------------------------------------------
    //
    // Remove (no-throw).
    //
    //----------------------------------------------------------------

    if (removeCount)
    {
        data.erase(data.begin(), data.begin() + removeCount);
        count -= removeCount;
    }
}

//================================================================
//
// BasicBuffer
//
//================================================================

template <typename Type>
using BasicBuffer = BasicBufferGreedy<Type>;

//================================================================
//
// absorbSeq
//
// May throw!
//
//================================================================

template <typename Type, typename Iterator, typename MoveOp>
void absorbSeq(BasicBuffer<Type>& buffer, Iterator newPtr, size_t newCount, size_t limit, const MoveOp& moveOp)
{
    if (newCount == 0)
        return;

    //----------------------------------------------------------------
    //
    // The new count can't be more than the limit.
    // Make newCount <= limit.
    //
    //----------------------------------------------------------------

    if (newCount > limit)
    {
        auto dropCount = newCount - limit;
        newPtr += dropCount;
        newCount -= dropCount;
    }

    //----------------------------------------------------------------
    //
    // Drop old items which are pushed out and add space
    // for new items (may throw).
    //
    //----------------------------------------------------------------

    buffer.cutToLimit(limit); // If the following operation fails, still limit it.

    ////

    auto savedCount = minv(limit - newCount, buffer.size());
    auto removedCount = buffer.size() - savedCount;

    ////

    buffer.removeFromBeginningAndAppendAtEnd(removedCount, newCount);

    //----------------------------------------------------------------
    //
    // Fill new items.
    //
    //----------------------------------------------------------------

    REMEMBER_CLEANUP_EX(newItemsRollback, buffer.removeFromEnd(newCount));

    ////

    auto dstPtr = buffer.end() - newCount;

    for_count (i, newCount)
        moveOp(*dstPtr++, *newPtr++);

    ////

    newItemsRollback.cancel();
}

//================================================================
//
// MsgRecord
//
//================================================================

struct MsgRecordData
{
    StringStorageEx<CharType> text;
    MsgKind kind;
    TimeMoment moment;
};

//----------------------------------------------------------------

struct MsgRecord : private MsgRecordData
{
    sysinline MsgRecord() =default;

    ////

    sysinline MsgRecord& operator=(const MsgRecord& that) =delete;
    sysinline MsgRecord(const MsgRecord& that) =delete;

    sysinline MsgRecord(MsgRecord&& that) =delete;

    ////

    sysinline MsgRecord& operator=(MsgRecord&& that) // won't clear "that"
    {
        kind = that.kind;
        moment = that.moment;
        text.swap(that.text);
        return *this;
    }

    sysinline const MsgRecordData& get() const {return *this;}

    ////

    sysinline void setKind(const MsgKind& value)
    {
        kind = value;
    }

    sysinline void setMoment(const TimeMoment& value)
    {
        moment = value;
    }

    sysinline void setText(const CharArray& value)
    {
        text = value;
    }
};

//================================================================
//
// LogBufferImpl
//
//================================================================

struct LogBufferImpl : public LogBuffer
{

    //----------------------------------------------------------------
    //
    // Config API.
    //
    //----------------------------------------------------------------

    virtual void setHistoryLimit(size_t value)
    {
        historyLimit = value;

        buffer.cutToLimit(historyLimit);
    }

    //----------------------------------------------------------------
    //
    // Clear memory.
    //
    //----------------------------------------------------------------

    virtual void clearMemory()
    {
        reset();

        buffer.clearMemory();
    }

    //----------------------------------------------------------------
    //
    // Buffer API.
    //
    //----------------------------------------------------------------

    //
    // Any update? (includes clearing)
    //

    virtual bool hasUpdates() const
    {
        return
            logCleared ||
            buffer.size() != 0 ||
            error != Error::None;
    }

    //
    // Reset the buffer to default state (except config).
    //

    void reset()
    {
        logCleared = false;
        buffer.clear(); // no-throw
        error = Error::None;
        lastModification.destroy();
    }

    ////

    virtual bool absorb(LogBuffer& other);

    virtual void moveFrom(LogBuffer& other);

    //----------------------------------------------------------------
    //
    // User API.
    //
    //----------------------------------------------------------------

    virtual void clearLog()
    {
        logCleared = true;
        buffer.clear();
        error = Error::None;
    }

    ////

    virtual void addMessage(const CharArray& text, MsgKind kind, const TimeMoment& moment);

    ////

    virtual size_t messageCount() const {return buffer.size();}

    ////

    virtual void refreshAllMoments(const TimeMoment& moment)
    {
        for (auto& r: buffer)
            r.setMoment(moment);

        lastModification = moment;
    }

    ////

    void readLastMessages(LogBufferReceiver& receiver, size_t count);
    void readFirstMessagesShowOverflow(LogBufferReceiver& receiver, size_t count);

    ////

    virtual OptionalObject<TimeMoment> getLastModification()
    {
        return lastModification;
    }

    //----------------------------------------------------------------
    //
    // Try to convert the error state to a message at the end of the log.
    // May throw.
    //
    //----------------------------------------------------------------

    void absorbErrorState();

    //----------------------------------------------------------------
    //
    // Config state.
    //
    //----------------------------------------------------------------

    size_t historyLimit = typeMax<size_t>();

    //----------------------------------------------------------------
    //
    // The log update buffer may have up to three sequential parts:
    // * "Clear log" item.
    // * Message items.
    // * Error state item.
    //
    //----------------------------------------------------------------

    bool logCleared = false;

    BasicBuffer<MsgRecord> buffer;

    //----------------------------------------------------------------
    //
    // The last moment.
    //
    // Updated on any modification.
    //
    //----------------------------------------------------------------

    OptionalObject<TimeMoment> lastModification;

    //----------------------------------------------------------------
    //
    // Error state.
    //
    // When the error state is raised (and not yet converted to a message),
    // it's displayed as the virtual last message of the log.
    //
    //----------------------------------------------------------------

    enum class Error {None, Internal, Alloc};
    Error error = Error::None;

    //----------------------------------------------------------------
    //
    // Error message.
    //
    //----------------------------------------------------------------

    auto errorMessage() const
    {
        return error == Error::Alloc ?
            STR("Log allocation error. The log may be incomplete.") :
            STR("Log internal error. The log may be incomplete.");
    }
};

UniquePtr<LogBuffer> LogBuffer::create()
{
    return makeUnique<LogBufferImpl>();
}

//================================================================
//
// logExceptBegin
// logExceptEnd
//
// Catch exceptions and record them to the error state.
//
//================================================================

#define logExceptBegin \
    try {

#define logExceptEnd \
    } \
    catch (const bad_alloc&) \
    { \
        error = Error::Alloc; \
    } \
    catch (...) \
    { \
        error = Error::Internal; \
    }

//================================================================
//
// LogBufferImpl::absorbErrorState
//
// May throw!
//
//================================================================

void LogBufferImpl::absorbErrorState()
{
    if (error == Error::None)
        return;

    //
    // Add the message.
    //

    auto msg = errorMessage();

    THROW_REQUIRE(lastModification);

    ////

    auto moveOp = [&] (auto& dst, auto& src)
    {
        STRESS_TEST_POINT(rand() % 5 != 0)
        dst.setKind(msgErr);
        dst.setMoment(*lastModification);
        dst.setText(msg);
    };

    absorbSeq(buffer, (int*) 0, 1, historyLimit, moveOp);

    //
    // Success.
    //

    error = Error::None;
}

//================================================================
//
// LogBufferImpl::addMessage
//
//================================================================

void LogBufferImpl::addMessage(const CharArray& text, MsgKind kind, const TimeMoment& moment)
{
    logExceptBegin;

    lastModification = moment;

    //
    // Absorb error state before any growth.
    //

    absorbErrorState();

    //
    // Add the message.
    //

    auto moveOp = [&] (auto& dst, auto& src)
    {
        STRESS_TEST_POINT(rand() % 13 == 0)
        dst.setKind(kind);
        dst.setMoment(moment);
        dst.setText(text);
    };

    absorbSeq(buffer, (int*) 0, 1, historyLimit, moveOp);

    ////

    logExceptEnd;
}

//================================================================
//
// LogBufferImpl::readLastMessages
//
//================================================================

void LogBufferImpl::readLastMessages(LogBufferReceiver& receiver, size_t count)
{
    REMEMBER_CLEANUP_EX(displayError, receiver(errorMessage(), msgErr, *lastModification));
    displayError.cancel();

    if (error != Error::None && lastModification && count >= 1)
    {
        count--; // reserve the last message for the error state message
        displayError.activate();
    }

    ////

    count = clampMax(count, buffer.size());

    auto bufferPtr = buffer.end() - count;

    ////

    for_count (i, count)
    {
        auto& r = (*bufferPtr++).get();
        receiver(CharArray(r.text.data(), r.text.size()), r.kind, r.moment);
    }
}

//================================================================
//
// LogBufferImpl::readFirstMessagesShowOverflow
//
//================================================================

void LogBufferImpl::readFirstMessagesShowOverflow(LogBufferReceiver& receiver, size_t count)
{
    //
    // Error state?
    //

    REMEMBER_CLEANUP_EX(displayError, receiver(errorMessage(), msgErr, *lastModification));
    displayError.cancel();

    if (error != Error::None && lastModification && count >= 1)
    {
        count--;
        displayError.activate();
    }

    //
    // Complete log?
    //

    REMEMBER_CLEANUP_EX(displayOverflow, receiver(STR("... and more"), msgWarn, *lastModification));
    displayOverflow.cancel();

    if (buffer.size() > count && lastModification && count >= 1)
    {
        count--;
        displayOverflow.activate();
    }

    //
    // Messages.
    //

    count = clampMax(count, buffer.size());

    auto bufferPtr = buffer.begin();

    for_count (i, count)
    {
        auto& r = (*bufferPtr++).get();
        receiver(CharArray(r.text.data(), r.text.size()), r.kind, r.moment);
    }
}

//================================================================
//
// LogBufferImpl::absorb
//
//================================================================

bool LogBufferImpl::absorb(LogBuffer& other)
{
    logExceptBegin;

    auto& that = (LogBufferImpl&) other;

    //----------------------------------------------------------------
    //
    // Remember to clear the absorbed buffer in any case.
    // The absorption takes as much data as possible.
    //
    //----------------------------------------------------------------

    REMEMBER_CLEANUP(that.reset());

    //----------------------------------------------------------------
    //
    // Update the last modification.
    //
    //----------------------------------------------------------------

    if (that.lastModification)
        lastModification = *that.lastModification;

    //----------------------------------------------------------------
    //
    // Before any growth absorb the error state.
    //
    //----------------------------------------------------------------

    absorbErrorState();

    //----------------------------------------------------------------
    //
    // Absorb clearing update.
    //
    //----------------------------------------------------------------

    if (that.logCleared)
        clearLog();

    //----------------------------------------------------------------
    //
    // Absorb buffer updates.
    //
    //----------------------------------------------------------------

    if (that.buffer.size() == 0)
    {
    }
    else if (buffer.size() == 0 || that.buffer.size() >= historyLimit)
    {
        // Full content replacement.
        buffer.moveFrom(that.buffer);
        buffer.cutToLimit(historyLimit);
    }
    else
    {
        auto moveOp = [] (auto& dst, auto& src)
        {
            dst = std::move(src);
        };

        absorbSeq(buffer, that.buffer.begin(), that.buffer.size(), historyLimit, moveOp);
    }

    //----------------------------------------------------------------
    //
    // Absorb error state.
    //
    //----------------------------------------------------------------

    if (that.error != Error::None)
    {
        error = that.error;
        lastModification = that.lastModification;

        absorbErrorState();
    }

    ////

    logExceptEnd;

    return true; // Errors are handled internally
}

//================================================================
//
// LogBufferImpl::moveFrom
//
//================================================================

void LogBufferImpl::moveFrom(LogBuffer& other)
{
    logExceptBegin; // Body shoudn't throw, but.

    auto& that = (LogBufferImpl&) other;

    REMEMBER_CLEANUP(that.reset());

    ////

    this->logCleared = that.logCleared;
    this->buffer.moveFrom(that.buffer); // no-throw
    this->lastModification = that.lastModification;
    this->error = that.error;

    ////

    logExceptEnd;
}
