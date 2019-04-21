#pragma once

#include "dataAlloc/arrayObjMemStatic.h"
#include "dataAlloc/arrayObjMem.h"
#include "numbers/int/intType.h"

//================================================================
//
// HistoryState
//
//================================================================

class HistoryState
{

    template <typename T>
    friend class HistoryGeneric;

private:

    Space storedEnd = 0;
    Space storedCount = 0;

};

//================================================================
//
// HistoryRanges
//
//================================================================

template <typename Type>
struct HistoryRanges
{
    Array<Type> a, b;
};

//================================================================
//
// HistoryObj
//
//================================================================

template <typename ContainerArray>
class HistoryGeneric
{

public:

    using Type = typename ContainerArray::Type;

    //----------------------------------------------------------------
    //
    // Reallocation.
    //
    //----------------------------------------------------------------

    template <typename Kit>
    stdbool realloc(Space size, stdPars(Kit))
    {
        dealloc(); // also resets queue
        return buffer.realloc(size, stdPassThru);
    }

    bool reallocStatic(Space size)
    {
        dealloc(); // also resets queue
        return buffer.reallocStatic(size);
    }

    void dealloc()
    {
        buffer.dealloc();
        storedEnd = 0;
        storedCount = 0;
    }

    sysinline Space allocSize() const {return buffer.size();}

    //----------------------------------------------------------------
    //
    // Operations.
    //
    //----------------------------------------------------------------

public:

    sysinline Space size() const
    {
        return storedCount;
    }

    ////

    sysinline void clear()
    {
        storedEnd = 0;
        storedCount = 0;
    }

    ////

    sysinline void removeOldest()
    {
        storedCount = clampMin(storedCount-1, 0);
    }

    ////

    sysinline void shorten(Space keepCount)
    {
        storedCount = clampMax(storedCount, keepCount);
    }

    ////

    sysinline void rollback(Space count)
    {
        ARRAY_EXPOSE(buffer);

        count = clampRange(count, 0, storedCount); // [0, storedCount]
        storedCount -= count;

        // storedEnd was [0, bufferSize)
        storedEnd -= count; // [-bufferSize, bufferSize), max storedCount=bufferSize

        if (storedEnd < 0)
            storedEnd += bufferSize; // [0, bufferSize)
    }

    //----------------------------------------------------------------
    //
    // addLocation: Get a pointer to the element to be reused.
    // addAdvance: Advance the queue.
    // add: Get a pointer to the element to be reused and advance the queue.
    //
    //----------------------------------------------------------------

public:

    sysinline Type* addLocation()
    {
        Type* result = 0;

        ARRAY_EXPOSE(buffer);

        if (storedEnd >= 0 && storedEnd < bufferSize)
            result = &helpModify(bufferPtr[storedEnd]);

        return result;
    }

    ////

    sysinline void addAdvance()
    {
        ARRAY_EXPOSE(buffer);
        requirev(storedEnd >= 0 && storedEnd < bufferSize);

        if (++storedEnd == bufferSize)
            storedEnd = 0;

        storedCount = clampMax(storedCount + 1, bufferSize);
    }

    ////

    sysinline Type* add()
    {
        Type* result = addLocation();
        addAdvance();
        return result;
    }

    //----------------------------------------------------------------
    //
    // State manipulation.
    //
    //----------------------------------------------------------------

public:

    sysinline void saveState(HistoryState& state)
    {
        state.storedCount = storedCount;
        state.storedEnd = storedEnd;
    }

    sysinline void restoreState(const HistoryState& state)
    {
        ARRAY_EXPOSE(buffer);
        storedCount = clampRange<Space>(state.storedCount, 0, bufferSize);
        storedEnd = clampRange<Space>(state.storedEnd, 0, clampMin(bufferSize-1, 0));
    }

    //----------------------------------------------------------------
    //
    // Element access.
    //
    //----------------------------------------------------------------

public:

    sysinline Type* operator []
    (
        Space index // [0, size)
    )
    const
    {
        Type* result = 0;

        ARRAY_EXPOSE(buffer);
        require_ex(bufferSize != 0, nullptr);

        Space I = -index-1;
        require_ex(-storedCount <= I && I < 0, nullptr);

        Space relativeIndex = storedEnd + I; // only to the left, minimum -bufferSize
        if (relativeIndex < 0) relativeIndex += bufferSize; // [0, bufferSize - 1]

        require_ex(relativeIndex >= 0 && relativeIndex < bufferSize, nullptr);
        result = &helpModify(bufferPtr[relativeIndex]);

        return result;
    }

    //----------------------------------------------------------------
    //
    // getRanges
    //
    //----------------------------------------------------------------

public:

    sysinline void getRanges(Array<Type>& a, Array<Type>& b) const
    {
        ARRAY_EXPOSE(buffer);

        //
        // separately check that the whole functions works correctly for bufferSize == 0
        //
        // otherwise bufferSize >= 1 and storedEnd is in [0, bufferSize)
        //

        Space storedBegin = storedEnd - storedCount; // [-bufferSize, bufferSize)

        //
        // split onto negative part [storedBegin, 0) and positive part [0, storedEnd)
        //

        Space negSize = clampMin(-storedBegin, 0); // active if storedBegin in [-bufferSize, 0)
        Space negOrg = storedBegin + bufferSize; // if active, range [0, bufferSize)
        buffer.subs(negOrg, negSize, a);

        ////

        Space posOrg = clampMin(storedBegin, 0);
        Space posEnd = storedEnd;
        buffer.subr(posOrg, posEnd, b);
    }

public:

    sysinline HistoryRanges<Type> getRanges() const
    {
        Array<Type> a, b;
        getRanges(a, b);
        return HistoryRanges<Type>{a, b};
    }

private:

    //
    // storage, index range [0, bufferSize)
    //

    ContainerArray buffer;

    //
    // the index of beyond-the-last element of stored sequence, range [0, bufferSize)
    // (if bufferSize == 0, which can happen on unsuccessful creation, the index is also 0)
    //

    Space storedEnd = 0;

    //
    // the number of stored elements, range [0, bufferSize]
    //

    Space storedCount = 0;

};

//----------------------------------------------------------------

template <typename Type, typename Action>
sysinline void historyForEach(const HistoryRanges<Type>& historyRanges, const Action& action)
{
    ARRAY_EXPOSE_EX(historyRanges.a, a);
    ARRAY_EXPOSE_EX(historyRanges.b, b);

    for (Space i = 0; i < aSize; ++i)
        action(aPtr[i]);

    for (Space i = 0; i < bSize; ++i)
        action(bPtr[i]);
}

//================================================================
//
// HistoryObjStatic
//
//================================================================

template <typename Type, Space maxSize>
class HistoryObjStatic : public HistoryGeneric< ArrayObjMemStatic<Type, maxSize> >
{
};

//================================================================
//
// HistoryObj
//
//================================================================

template <typename Type>
class HistoryObj : public HistoryGeneric< ArrayObjMem<Type> >
{
};
