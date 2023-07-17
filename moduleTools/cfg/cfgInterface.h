#pragma once

#include "charType/charType.h"
#include "cfg/cfgTextSerialization.h"
#include "cfg/cfgInterfaceFwd.h"
#include "storage/adapters/lambdaThunk.h"
#include "storage/opaqueStruct.h"

//================================================================
//
// CfgSerializeVariable
//
// Abstract serialization interface of a configuration variable.
//
//================================================================

struct CfgSerializeVariable
{
    // Synced flag. Should be reset when the variable is changed.
    virtual bool synced() const =0;
    virtual void setSynced(bool value) const =0;

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
// CfgVisitVar
//
//================================================================

struct CfgVisitVar
{
    virtual void operator()(const CfgSerializeVariable& var) const =0;
};

////

LAMBDA_THUNK
(
    cfgVisitVar,
    CfgVisitVar,
    void operator()(const CfgSerializeVariable& var) const,
    lambda(var)
)

////

struct CfgVisitVarNull : public CfgVisitVar
{
    virtual void operator()(const CfgSerializeVariable& var) const {}
};

//================================================================
//
// CfgVisitSignal
//
//================================================================

struct CfgVisitSignal
{
    virtual void operator()(const CfgSerializeSignal& signal) const =0;
};

////

LAMBDA_THUNK
(
    cfgVisitSignal,
    CfgVisitSignal,
    void operator()(const CfgSerializeSignal& signal) const,
    lambda(signal)
)

////

struct CfgVisitSignalNull : public CfgVisitSignal
{
    virtual void operator()(const CfgSerializeSignal& signal) const {}
};

//================================================================
//
// CfgScopeVisitor
//
//================================================================

struct CfgScopeVisitor
{
    virtual void enter(CfgScopeContext& context, const CharArray& name) const =0;
    virtual void leave(CfgScopeContext& context) const =0;
};

////

LAMBDA_THUNK2
(
    cfgScopeVisitor,
    CfgScopeVisitor,
    void enter(CfgScopeContext& context, const CharArray& name) const,
    lambda0(context, name),
    void leave(CfgScopeContext& context) const,
    lambda1(context)
)

////

struct CfgScopeVisitorNull : public CfgScopeVisitor
{
    virtual void enter(CfgScopeContext& context, const CharArray& name) const {}
    virtual void leave(CfgScopeContext& context) const {}
};

//================================================================
//
// CfgScopeVisitorCaller
//
//================================================================

class CfgScopeVisitorCaller
{

public:

    sysinline CfgScopeVisitorCaller
    (
        const CfgScopeVisitor& visitor,
        CfgScopeContext& context,
        const CharArray& name
    )
        :
        visitor{visitor},
        context{context}
    {
        visitor.enter(context, name);
    }

    sysinline ~CfgScopeVisitorCaller()
    {
        visitor.leave(context);
    }

private:

    const CfgScopeVisitor& visitor;
    CfgScopeContext& context;

};

//================================================================
//
// CFG_NAMESPACE*
//
//================================================================

#define CFG_NAMESPACE(name) \
    CFG_NAMESPACE_EX(STR(name))

#define CFG_NAMESPACE_EX(newName) \
    CfgScopeContext __cfgScopeContext; \
    CharArray __cfgScopeName = (newName); \
    CfgScopeVisitorCaller __cfgScopeVisitorCaller{kit.scopeVisitor, __cfgScopeContext, __cfgScopeName};
