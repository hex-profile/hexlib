#pragma once

#include "cfgTools/standardSignal.h"
#include "cfgTools/numericVar.h"
#include "cfgTools/overlayTakeover.h"
#include "prepTools/prepFor.h"
#include "prepTools/prepIncDec.h"

//================================================================
//
// MultiSwitch user class
//
// Implements integer selector for N positions activated by N specific signals.
// The user specifies N as class parameter and provides a signals' description array in serialization.
//
//================================================================

template <typename EnumType, EnumType positionCount, EnumType defaultPos>
class MultiSwitch;

//================================================================
//
// NameKeyCommentStruct
//
//================================================================

struct NameKeyCommentStruct
{
    CharArray name;
    CharArray key;
    CharArray comment;
};

//================================================================
//
// NameKeyComment
//
//================================================================

struct NameKeyComment : public NameKeyCommentStruct
{
    inline NameKeyComment(const CharArray& name, const CharArray& key = STR(""), const CharArray& comment = STR(""))
    {
        this->name = name;
        this->key = key;
        this->comment = comment;
    }
};

//================================================================
//
// serializeMultiSwitch
//
// Returns true if the switch is NOT changed.
//
//================================================================

bool serializeMultiSwitch
(
    const CfgSerializeKit& kit,
    const CharArray& name,
    NumericVar<size_t>& value,
    size_t descSize,
    StandardSignal signals[],
    const NameKeyCommentStruct descPtr[],
    bool cfgValuePrefix,
    bool signalPrefix
);

//================================================================
//
// Convenient serialize function
//
//================================================================

struct NameKeyPair
{
    NameKeyPair(const CharArray& name, const CharArray& key = STR(""))
        : name(name), key(key) {}

    CharArray name;
    CharArray key;
};

//----------------------------------------------------------------

#define SWSER_DEFINE \
    PREP_FOR1(16, SWSER_AUX0, _)

#define SWSER_AUX0(n, _) \
    SWSER_AUX1(PREP_INC(n))

#define SWSER_AUX1(n) \
    SWSER_MAKE_FUNC(n)

//----------------------------------------------------------------

#define SWSER_ENUM_PARAM(k, _) \
    const NameKeyPair& desc##k,

#define SWSER_PASS_PARAM(k, _) \
    {desc##k.name, desc##k.key, STR("")},

#define SWSER_MAKE_FUNC(n) \
    template <typename Kit> \
    inline bool serialize \
    ( \
        const Kit& kit, \
        const CharArray& name, \
        PREP_FOR0(n, SWSER_ENUM_PARAM, _) \
        bool cfgPrefix = true, \
        bool signalPrefix = true \
    ) \
    { \
        const NameKeyCommentStruct descArray[] = {PREP_FOR0(n, SWSER_PASS_PARAM, _)}; \
        COMPILE_ASSERT(positionCount == n); \
        return serialize(kit, name, descArray, cfgPrefix, signalPrefix); \
    }

//================================================================
//
// MultiSwitch
//
//================================================================

template <typename EnumType, EnumType positionCount, EnumType defaultPos>
class MultiSwitch
{

    COMPILE_ASSERT(positionCount >= 1 && 0 <= defaultPos && defaultPos <= positionCount - 1);

public:

    inline operator EnumType() const {return (EnumType) value();}
    inline EnumType operator()() const {return (EnumType) value();}

    void operator =(EnumType v)
        {value = v;}

public:

    inline bool serialize(const CfgSerializeKit& kit, const CharArray& name, const NameKeyCommentStruct descArray[], bool cfgPrefix = true, bool signalPrefix = true)
        {return serializeMultiSwitch(kit, name, value, positionCount, signals, descArray, cfgPrefix, signalPrefix);}

    SWSER_DEFINE

private:

    NumericVarStatic<size_t, 0, positionCount-1, defaultPos> value;
    StandardSignal signals[positionCount];

};

//================================================================
//
// ExclusiveMultiSwitch
//
// A multi-switch with exclusive resource takeover.
//
//================================================================

template <typename EnumType, EnumType positionCount, uint32 baseID>
class ExclusiveMultiSwitch
{

    COMPILE_ASSERT(positionCount >= 1);

    using Base = MultiSwitch<EnumType, positionCount, EnumType(0)>;

    //
    // serialize
    //

public:

    template <typename Kit>
    inline bool serialize
    (
        const Kit& kit,
        const CharArray& name,
        const NameKeyCommentStruct descArray[],
        bool cfgPrefix = true,
        bool signalPrefix = true
    )
    {
        //
        // Check if somebody has taken over the control
        //

        uint32 id = kit.overlayTakeover.getActiveID() - baseID;

        if_not (id >= 0 && id < positionCount)
            base = EnumType(0);

        //
        // Apply changes
        //

        EnumType prevValue = base;
        base.serialize(kit, name, descArray, cfgPrefix, signalPrefix);

        //
        // Take control if changed
        //

        if (base != prevValue && base != 0) // changed?
            kit.overlayTakeover.setActiveID(baseID + base());

        return (base == prevValue);
    }

    SWSER_DEFINE

    //
    // operator()
    //

    inline operator EnumType() const {return base;}
    inline EnumType operator()() const {return base;}

private:

    Base base;

};
