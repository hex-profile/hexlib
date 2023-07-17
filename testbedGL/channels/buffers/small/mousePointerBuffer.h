#pragma once

#include "storage/smartPtr.h"
#include "storage/optionalObject.h"
#include "point/pointBase.h"
#include "numbers/float/floatBase.h"

namespace mousePointerBuffer {

//================================================================
//
// MousePointer
//
//================================================================

struct MousePointer
{
    bool hasUpdates() const {return position || button0 || button1;}

    OptionalObject<Point<float32>> position;
    OptionalObject<bool> button0;
    OptionalObject<bool> button1;
};

//================================================================
//
// MousePointerBuffer
//
//================================================================

class MousePointerBuffer : private MousePointer
{

public:

    static UniquePtr<MousePointerBuffer> create()
        {return makeUnique<MousePointerBuffer>();}

    virtual ~MousePointerBuffer() {}

public:

    void clearMemory()
        {reset();}

    bool hasUpdates() const
        {return MousePointer::hasUpdates();}

    void reset()
        {set({});}

    ////

    bool absorb(MousePointerBuffer& that)
    {
        if (that.position)
            position = *that.position;

        if (that.button0)
            button0 = *that.button0;

        if (that.button1)
            button1 = *that.button1;

        that.reset();

        return true;
    }

    void moveFrom(MousePointerBuffer& that)
    {
        set(that.get());
        that.reset();
    }

    ////

    void set(const MousePointer& value)
    {
        MousePointer& the = *this;
        the = value;
    }

    MousePointer get() const
        {return *this;}

};

//----------------------------------------------------------------

}

using mousePointerBuffer::MousePointerBuffer;
