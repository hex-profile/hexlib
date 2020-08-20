#pragma once

#include "charType/charType.h"
#include "cfg/cfgTextSerialization.h"
#include "cfg/cfgInterfaceFwd.h"

//================================================================
//
// CfgNamespace
//
//================================================================

class CfgNamespace
{

public:

    CharArray desc;
    const CfgNamespace* prev;

public:

    inline CfgNamespace(const CharArray& desc, const CfgNamespace* prev)
        : desc(desc), prev(prev) {}

};

//----------------------------------------------------------------

#define CFG_NAMESPACE_EX(name) \
    CfgNamespace PREP_PASTE(newFrame, __LINE__)(name, kit.scope); \
    auto oldKit = kit; \
    auto kit = oldKit; \
    kit.scope = &PREP_PASTE(newFrame, __LINE__)

#define CFG_NAMESPACE(name) \
    CFG_NAMESPACE_EX(STR(name))

//================================================================
//
// CfgSerializeVariable
//
// Abstract serialization interface of a configuration variable.
//
//================================================================

struct CfgSerializeVariable
{

    // Dirty flag. Should be set internally when the variable is changed.
    virtual bool changed() const =0;
    virtual void clearChanged() const =0;

    // Set default variable value.
    virtual void resetValue() const =0;

    // Get the variable name.
    virtual bool getName(CfgOutputString& result) const =0;

    // Save the value to text format.
    virtual bool getTextValue(CfgWriteStream& s) const =0;

    // Load the value from text format.
    // If the function fails, the variable value should not be changed.
    virtual bool setTextValue(CfgReadStream& s) const =0;

    // Get the variable text comment.
    // Block comment can be multi-line.
    virtual bool getTextComment(CfgWriteStream& s) const =0;
    virtual bool getBlockComment(CfgWriteStream& s) const =0;

};

//================================================================
//
// CfgSerializeSignal
//
// Abstract serialization interface of a user signal.
//
//================================================================

struct CfgSerializeSignal
{
    // Get name.
    virtual bool getName(CfgOutputString& result) const =0;

    // Get accelerator key.
    virtual bool getKey(CfgOutputString& result) const =0;

    // Get text comment.
    virtual bool getTextComment(CfgOutputString& result) const =0;

    // Set the number of times signal has happened.
    virtual void setImpulseCount(int32 count) const =0;

    // Support for implementation.
    virtual void setID(uint32 id) const =0;
    virtual uint32 getID() const =0;
};

//================================================================
//
// CfgVisitor
//
//================================================================

struct CfgVisitor
{
    virtual void operator()(const CfgNamespace* scope, const CfgSerializeVariable& var) =0;
    virtual void operator()(const CfgNamespace* scope, const CfgSerializeSignal& signal) =0;
};
