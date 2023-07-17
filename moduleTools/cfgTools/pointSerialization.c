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
        ensure(cfgWrite(s, value.X));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.Y));
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
        ensure(cfgWrite(s, value.X));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.Y));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.Z));
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
        ensure(cfgWrite(s, value.X));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.Y));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.Z));
        ensure(cfgWrite(s, STR(", ")));
        ensure(cfgWrite(s, value.W));
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
// CfgRead<PointFamily>
//
//================================================================

template <>
struct CfgRead<PointFamily>
{
    template <typename PointType>
    static inline bool func(CfgReadStream& s, PointType& value)
    {
        PointType newValue;

        ensure(cfgRead(s, newValue.X));
        ensure(skipComma(s));
        ensure(cfgRead(s, newValue.Y));

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
        PointType newValue;

        ensure(cfgRead(s, newValue.X));
        ensure(skipComma(s));
        ensure(cfgRead(s, newValue.Y));
        ensure(skipComma(s));
        ensure(cfgRead(s, newValue.Z));

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
        PointType newValue;

        ensure(cfgRead(s, newValue.X));
        ensure(skipComma(s));
        ensure(cfgRead(s, newValue.Y));
        ensure(skipComma(s));
        ensure(cfgRead(s, newValue.Z));
        ensure(skipComma(s));
        ensure(cfgRead(s, newValue.W));

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
