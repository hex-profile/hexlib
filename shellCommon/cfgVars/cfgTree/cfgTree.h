#pragma once

#include "cfgVars/types/charTypes.h"
#include "storage/adapters/callable.h"
#include "storage/smartPtr.h"
#include "numbers/int/intBase.h"
#include "storage/optionalObject.h"

namespace cfgTree {

using namespace cfgVarsImpl;

//================================================================
//
// Hash
//
//================================================================

enum class Hash: uint32 {};

//================================================================
//
// getHash
//
//================================================================

Hash getHash(const StringRef& str);

//================================================================
//
// NameRef
//
//================================================================

struct NameRef
{
    Hash hash = getHash({});
    StringRef str;

    sysinline NameRef() =default;

    sysinline explicit NameRef(const StringRef& str)
        : hash{getHash(str)}, str{str} {}

    sysinline NameRef(Hash hash, const StringRef& str)
        : hash{hash}, str{str} {}
};

//================================================================
//
// NodeHandler
//
//================================================================

using NodeHandler = Callable<void (struct Node& node)>;

//================================================================
//
// GetDataResult
//
//================================================================

struct GetDataResult
{
    StringRef value;
    StringRef comment;
    StringRef blockComment;
};

using SetDataArgs = GetDataResult;

//================================================================
//
// Node
//
// Some methods of this class MAY throw exceptions, mostly std::bad_alloc.
//
//================================================================

struct Node
{
    //
    // Init/deinit.
    //

    static UniquePtr<Node> create() may_throw;
    virtual ~Node() {}

    ////

    virtual void clearAll() =0;
    virtual void dealloc() =0;

    //----------------------------------------------------------------
    //
    // Name.
    //
    //----------------------------------------------------------------

    virtual StringRef getName() const =0;
    virtual void clearName() =0;

    //----------------------------------------------------------------
    //
    // Data.
    //
    //----------------------------------------------------------------

    virtual bool hasData() const =0;

    virtual void clearData() =0;

    virtual GetDataResult getData() const =0;

    virtual void setData(const SetDataArgs& args) may_throw =0;

    // Slower, but updates the internal "changed" flag.
    virtual void setDataEx(const SetDataArgs& args) may_throw =0;

    //----------------------------------------------------------------
    //
    // Children.
    //
    //----------------------------------------------------------------

    virtual bool hasChildren() const =0;

    virtual void clearChildren() =0;

    virtual void forAllChildren(NodeHandler& handler) =0;

    //
    // May return NULL, never throws.
    //

    virtual Node* findChild(const NameRef& name) =0;

    //
    // Never returns NULL, but may throw.
    //

    virtual Node* findOrCreateChild(const NameRef& name) may_throw =0;

    //
    // Remove a child by name if it exists.
    //

    virtual bool removeChild(const NameRef& name) =0;

    //----------------------------------------------------------------
    //
    // Delta buffer API.
    //
    // No exceptions here.
    //
    //----------------------------------------------------------------

    bool hasUpdates() const
        {return hasData() || hasChildren();}

    sysinline void clearMemory()
        {dealloc();}

    sysinline void reset()
        {clearAll();}

    virtual void moveFrom(Node& other) =0;

    virtual bool absorb(Node& other) =0;

    //----------------------------------------------------------------
    //
    // Diagnostic.
    //
    //----------------------------------------------------------------

    virtual size_t allocatedBytes() const =0;

    //----------------------------------------------------------------
    //
    // Generating differential update.
    //
    // Optimally works for a small number of changed variables.
    //
    //----------------------------------------------------------------

    virtual void clearAllDataChangedFlags() =0;

    virtual void generateUpdate(Node& dst) may_throw =0;

};

//----------------------------------------------------------------

}

using CfgTree = cfgTree::Node;
