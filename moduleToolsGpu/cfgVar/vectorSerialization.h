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
        ensure(cfgWrite(s, value.x));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.y));
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
        ensure(cfgWrite(s, value.x));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.y));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.z));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.w));
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

    ensure(s.readChars(&tmp, 1));

    if_not (tmp == ' ')
        ensure(s.unreadChar());

    ////

    ensure(s.readChars(&tmp, 1));
    ensure(tmp == ',');

    ////

    ensure(s.readChars(&tmp, 1));

    if_not (tmp == ' ')
        ensure(s.unreadChar());

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

        ensure(cfgRead(s, newValue.x));
        ensure(readVectorSeparator(s));
        ensure(cfgRead(s, newValue.y));

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

        ensure(cfgRead(s, newValue.x));
        ensure(readVectorSeparator(s));

        ensure(cfgRead(s, newValue.y));
        ensure(readVectorSeparator(s));

        ensure(cfgRead(s, newValue.z));
        ensure(readVectorSeparator(s));

        ensure(cfgRead(s, newValue.w));

        value = newValue;

        return true;
    }
};
