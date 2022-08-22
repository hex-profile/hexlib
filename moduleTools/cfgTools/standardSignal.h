#pragma once

#include "cfg/cfgInterface.h"

//================================================================
//
// StandardSignal
//
// Standard signal implementation.
//
//================================================================

class StandardSignal
{

public:

    inline operator int32() const
        {return impulseCount;}

    inline int32 operator() () const
        {return impulseCount;}

public:

    inline void clear() {impulseCount = 0;}

public:

    // Returns true if NOT signaled
    inline bool serialize
    (
        const CfgSerializeKit& kit,
        const CharArray& name,
        const CharArray& key = STR(""),
        const CharArray& comment = STR("")
    );

private:

    friend class SerializeStandardSignal;

    int32 impulseCount = 0;
    uint32 internalId = 0;

};

//================================================================
//
// SerializeStandardSignal
//
// Simple serializer: Name, key and comment are specified
// as C strings right in the process of serialization.
//
//================================================================

class SerializeStandardSignal : public CfgSerializeSignal
{

public:

    inline SerializeStandardSignal
    (
        StandardSignal& baseSignal,
        const CharArray& name,
        const CharArray& key,
        const CharArray& comment
    )
        :
        baseSignal(baseSignal),
        name(name),
        key(key),
        comment(comment)
    {
    }

    virtual bool getName(CfgOutputString& result) const
        {return result.addStr(name);}

    virtual bool getKey(CfgOutputString& result) const
        {return result.addStr(key);}

    virtual bool getTextComment(CfgOutputString& result) const
        {return result.addStr(comment);}

    virtual void setImpulseCount(int32 count) const
        {baseSignal.impulseCount = count;}

    virtual void setID(uint32 id) const
        {baseSignal.internalId = id;}

    virtual uint32 getID() const
        {return baseSignal.internalId;}

private:

    StandardSignal& baseSignal;
    const CharArray name;
    const CharArray key;
    const CharArray comment;

};

//================================================================
//
// StandardSignal::serialize
//
//================================================================

inline bool StandardSignal::serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& key, const CharArray& comment)
{
    SerializeStandardSignal serializeSignal(*this, name, key, comment);
    kit.visitor(kit.scope, serializeSignal);
    return (impulseCount == 0);
}
