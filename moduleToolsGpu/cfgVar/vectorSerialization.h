#pragma once

#include "cfg/cfgTextSerialization.h"
#include "vectorTypes/vectorType.h"

//================================================================
//
// CfgWrite<VectorX2>
//
//================================================================

template <>
struct CfgWrite<VectorX2>
{
    template <typename VectorType>
    static inline bool func(CfgWriteStream& s, const VectorType& value)
    {
        require(cfgWrite(s, value.x));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.y));
        return true;
    }
};

//================================================================
//
// CfgWrite<VectorX4>
//
//================================================================

template <>
struct CfgWrite<VectorX4>
{
    template <typename VectorType>
    static inline bool func(CfgWriteStream& s, const VectorType& value)
    {
        require(cfgWrite(s, value.x));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.y));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.z));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.w));
        return true;
    }
};

//================================================================
//
// readVectorSeparator
//
//================================================================

inline bool readVectorSeparator(CfgReadStream& s)
{
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

    return true;
}

//================================================================
//
// CfgRead<VectorX2>
//
//================================================================

template <>
struct CfgRead<VectorX2>
{
    template <typename VectorType>
    static bool func(CfgReadStream& s, VectorType& value)
    {
        VectorType newValue(value);

        require(cfgRead(s, newValue.x));
        require(readVectorSeparator(s));
        require(cfgRead(s, newValue.y));

        value = newValue;

        return true;
    }
};

//================================================================
//
// CfgRead<VectorX4>
//
//================================================================

template <>
struct CfgRead<VectorX4>
{
    template <typename VectorType>
    static bool func(CfgReadStream& s, VectorType& value)
    {
        VectorType newValue(value);

        require(cfgRead(s, newValue.x));
        require(readVectorSeparator(s));

        require(cfgRead(s, newValue.y));
        require(readVectorSeparator(s));

        require(cfgRead(s, newValue.z));
        require(readVectorSeparator(s));

        require(cfgRead(s, newValue.w));

        value = newValue;

        return true;
    }
};
