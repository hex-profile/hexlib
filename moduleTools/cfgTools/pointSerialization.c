#include "numericVar.inl"

#include "cfg/cfgTextSerialization.h"

#include "point/point.h"
#include "point3d/point3d.h"
#include "point4d/point4d.h"

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
// CfgWrite<Point3DFamily>
//
//================================================================

template <>
struct CfgWrite<Point3DFamily>
{
    template <typename PointType>
    static inline bool func(CfgWriteStream& s, const PointType& value)
    {
        require(cfgWrite(s, value.X));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.Y));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.Z));
        return true;
    }
};

//================================================================
//
// CfgWrite<Point4DFamily>
//
//================================================================

template <>
struct CfgWrite<Point4DFamily>
{
    template <typename PointType>
    static inline bool func(CfgWriteStream& s, const PointType& value)
    {
        require(cfgWrite(s, value.X));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.Y));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.Z));
        require(cfgWrite(s, STR(", ")));
        require(cfgWrite(s, value.W));
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
// CfgRead<PointFamily>
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
        require(skipComma(s));
        require(cfgRead(s, newValue.Y));

        value = newValue;

        return true;
    }
};

//================================================================
//
// CfgRead<Point3DFamily>
//
//================================================================

template <>
struct CfgRead<Point3DFamily>
{
    template <typename PointType>
    static inline bool func(CfgReadStream& s, PointType& value)
    {
        PointType newValue(value);

        require(cfgRead(s, newValue.X));
        require(skipComma(s));
        require(cfgRead(s, newValue.Y));
        require(skipComma(s));
        require(cfgRead(s, newValue.Z));

        value = newValue;

        return true;
    }
};

//================================================================
//
// CfgRead<Point4DFamily>
//
//================================================================

template <>
struct CfgRead<Point4DFamily>
{
    template <typename PointType>
    static inline bool func(CfgReadStream& s, PointType& value)
    {
        PointType newValue(value);

        require(cfgRead(s, newValue.X));
        require(skipComma(s));
        require(cfgRead(s, newValue.Y));
        require(skipComma(s));
        require(cfgRead(s, newValue.Z));
        require(skipComma(s));
        require(cfgRead(s, newValue.W));

        value = newValue;

        return true;
    }
};

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class SerializeNumericVar<Point<Type>>; \
    template class SerializeNumericVar<Point3D<Type>>; \
    template class SerializeNumericVar<Point4D<Type>>;

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO