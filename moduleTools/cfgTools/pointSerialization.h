#pragma once

#include "cfg/cfgTextSerialization.h"
#include "point/point.h"

//================================================================
//
// CfgWrite<PointFamily>
//
//================================================================

template <>
struct CfgWrite<PointFamily>
{
    template <typename PointType>
    static inline bool func(CfgWriteStream& s, const PointType& value)
    {
        require(cfgWrite(s, value.X));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.Y));
        return true;
    }
};

//================================================================
//
// CfgRead
//
//================================================================

template <>
struct CfgRead<PointFamily>
{
    template <typename PointType>
    static inline bool func(CfgReadStream& s, PointType& value)
    {
        PointType newValue(value);

        require(cfgRead(s, newValue.X));

        ////

        CharType tmp(0);

        require(s.readChars(&tmp, 1));

        if_not (tmp == ' ')
            require(s.unreadChar());

        ////

        require(s.readChars(&tmp, 1));
        require(tmp == ',');

        ////

        require(s.readChars(&tmp, 1));

        if_not (tmp == ' ')
            require(s.unreadChar());

        ////

        require(cfgRead(s, newValue.Y));

        value = newValue;

        return true;
    }
};
