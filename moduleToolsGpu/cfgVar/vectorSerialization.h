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
// skipComma
//
//================================================================

inline bool skipComma(CfgReadStream& s)
{
    s.skipSpaces();
    ensure(s.skipText(STR(",")));
    s.skipSpaces();
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
        VectorType newValue;

        ensure(cfgRead(s, newValue.x));
        ensure(skipComma(s));
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
        VectorType newValue;

        ensure(cfgRead(s, newValue.x));
        ensure(skipComma(s));

        ensure(cfgRead(s, newValue.y));
        ensure(skipComma(s));

        ensure(cfgRead(s, newValue.z));
        ensure(skipComma(s));

        ensure(cfgRead(s, newValue.w));

        value = newValue;

        return true;
    }
};
