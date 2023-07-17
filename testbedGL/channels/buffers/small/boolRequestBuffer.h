#pragma once

#include "storage/smartPtr.h"

//================================================================
//
// BoolRequestBuffer
//
//================================================================

template <unsigned hash>
class BoolRequestBuffer
{

public:

    static UniquePtr<BoolRequestBuffer> create()
        {return makeUnique<BoolRequestBuffer>();}

    virtual ~BoolRequestBuffer() {}

public:

    void clearMemory()
        {request = false;}

    bool hasUpdates() const
        {return request;}

    void reset()
        {request = false;}

    bool absorb(BoolRequestBuffer& that)
    {
        request |= that.request;
        that.request = false;
        return true;
    }

    void moveFrom(BoolRequestBuffer& that)
    {
        request = that.request;
        that.request = false;
    }

    void addRequest()
    {
        request = true;
    }

private:

    bool request = false;

};
