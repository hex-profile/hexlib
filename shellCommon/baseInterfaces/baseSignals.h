#pragma once

#include "numbers/int/intBase.h"
#include "charType/charType.h"
#include "data/array.h"
#include "point/pointBase.h"

//================================================================
//
// BaseActionId
//
//================================================================

using BaseActionId = uint32;

//================================================================
//
// Base actions.
//
//================================================================

namespace baseActionId
{
    constexpr BaseActionId MouseLeftDown = 0xFFFFFFFEu;
    constexpr BaseActionId MouseLeftUp = 0xFFFFFFFDu;

    constexpr BaseActionId MouseRightDown = 0xFFFFFFFCu;
    constexpr BaseActionId MouseRightUp = 0xFFFFFFFBu;

    constexpr BaseActionId WheelDown = 0xFFFFFFFAu;
    constexpr BaseActionId WheelUp = 0xFFFFFFF9u;

    constexpr BaseActionId SaveConfig = 0xFFFFFFF8u;
    constexpr BaseActionId LoadConfig = 0xFFFFFFF7u;
    constexpr BaseActionId EditConfig = 0xFFFFFFF6u;

    constexpr BaseActionId ResetupActions = 0xFFFFFFF5u;
}

//================================================================
//
// BaseActionSetup
//
//================================================================

struct BaseActionSetup
{
    virtual bool actsetClear() =0;
    virtual bool actsetUpdate() =0;
    virtual bool actsetAdd(BaseActionId id, const CharType* key, const CharType* name, const CharType* comment) =0;
};

//================================================================
//
// BaseMousePos
//
//================================================================

class BaseMousePos
{

public:

    BaseMousePos() =default;

    inline BaseMousePos(const Point<Space>& pos) 
        : posX(pos.X), posY(pos.Y) {}

    inline bool valid() const
        {return posX >= 0 && posY >= 0;}

    inline Point<Space> pos()
        {return Point<Space>{posX, posY};}

private:

    int16 posX = -1;
    int16 posY = -1;

};

//================================================================
//
// ActionRec
//
//================================================================

struct BaseActionRec
{
    BaseActionId id;
    BaseMousePos mousePos;
};

//================================================================
//
// BaseActionReceiving
//
// Gets actions that happened from the previous action receiving.
// Actions are 'transferred' to the client.
//
//================================================================

struct BaseActionReceiver
{
    virtual void process(const Array<const BaseActionRec>& actions) =0;
};

//----------------------------------------------------------------

struct BaseActionReceiving
{
    virtual void getActions(BaseActionReceiver& receiver) =0;
};
