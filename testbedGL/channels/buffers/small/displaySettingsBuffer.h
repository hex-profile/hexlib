#pragma once

#include "storage/smartPtr.h"
#include "storage/optionalObject.h"
#include "data/space.h"
#include "point/point.h"

namespace displaySettingsBuffer {

//================================================================
//
// DisplaySettings
//
//================================================================

struct DisplaySettings
{
    OptionalObject<Point<Space>> desiredOutputSize;
};

//================================================================
//
// DisplaySettingsBuffer
//
//================================================================

class DisplaySettingsBuffer
{

public:

    static UniquePtr<DisplaySettingsBuffer> create()
        {return makeUnique<DisplaySettingsBuffer>();}

    virtual ~DisplaySettingsBuffer() {}

public:

    void clearMemory()
        {reset();}

    bool hasUpdates() const
        {return !!data.desiredOutputSize;}

    void reset()
        {data.desiredOutputSize = {};}

    ////

    bool absorb(DisplaySettingsBuffer& that)
    {
        if (that.data.desiredOutputSize)
            data.desiredOutputSize = *that.data.desiredOutputSize;

        that.reset();

        return true;
    }

    void moveFrom(DisplaySettingsBuffer& that)
    {
        data = that.data;
        that.reset();
    }

    ////

    void set(const DisplaySettings& value)
        {data = value;}

    DisplaySettings get() const
        {return data;}


private:

    DisplaySettings data;

};

//----------------------------------------------------------------

}

using displaySettingsBuffer::DisplaySettingsBuffer;
