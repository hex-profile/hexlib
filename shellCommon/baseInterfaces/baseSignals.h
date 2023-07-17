#pragma once

#include "numbers/int/intBase.h"
#include "charType/charType.h"
#include "data/array.h"
#include "point/pointBase.h"
#include "baseInterfaces/actionDefs.h"

//================================================================
//
// Legacy predefined action IDs, compatible with AT shell
// and Debug Bridge API.
//
//================================================================

namespace baseActionId
{
    constexpr ActionId MouseLeftDown = 0xFFFFFFFEu;
    constexpr ActionId MouseLeftUp = 0xFFFFFFFDu;

    constexpr ActionId MouseRightDown = 0xFFFFFFFCu;
    constexpr ActionId MouseRightUp = 0xFFFFFFFBu;

    constexpr ActionId WheelDown = 0xFFFFFFFAu;
    constexpr ActionId WheelUp = 0xFFFFFFF9u;

    constexpr ActionId SaveConfig = 0xFFFFFFF8u;
    constexpr ActionId LoadConfig = 0xFFFFFFF7u;
    constexpr ActionId EditConfig = 0xFFFFFFF6u;

    constexpr ActionId ResetupActions = 0xFFFFFFF5u;

    constexpr ActionId LastPredefinedAction = ResetupActions;
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
    virtual bool actsetAdd(ActionId id, ActionKey key, const CharType* name, const CharType* comment) =0;
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
// BaseActionRec
//
//================================================================

struct BaseActionRec
{
    ActionId id;
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
