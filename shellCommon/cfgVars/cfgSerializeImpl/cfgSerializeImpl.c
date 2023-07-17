#include "cfgSerializeImpl.h"

#include "cfg/cfgInterface.h"
#include "cfgVars/types/stringStorage.h"
#include "errorLog/convertExceptions.h"
#include "parseTools/parseTools.h"
#include "parseTools/readTools.h"
#include "parseTools/writeTools.h"
#include "podVector/podVector.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "errorLog/errorLog.h"

namespace cfgSerializeImpl {

using namespace std;

//================================================================
//
// WriteStreamImpl
//
// Methods may throw exceptions!
//
//================================================================

class WriteStreamImpl : public CfgWriteStream
{

public:

    sysinline WriteStreamImpl() =default;

    sysinline WriteStreamImpl(StringStorage& vec)
        : vec{vec}
    {
    }

public:

    sysinline void clear()
        {vec.clear();}

public:

    sysinline auto size() const
        {return vec.size();}

    sysinline auto str() const
        {return vec.str();}

    //----------------------------------------------------------------
    //
    // writeChars
    //
    //----------------------------------------------------------------

public:

    sysinline void writeBuf(const Char* ptr, size_t size)
    {
        vec.append(ptr, size, true);
    }

    ////

    virtual bool writeChars(const Char* array, size_t size)
    {
        writeBuf(array, size);
        return true;
    }

    //----------------------------------------------------------------
    //
    // Integers.
    //
    //----------------------------------------------------------------

public:

    template <typename Int>
    sysinline bool writeInt(const Int& value)
    {
        auto writer = [&] (auto* ptr, auto size)
            {writeBuf(ptr, size);};

        ::writeInt<Char>(value, writer, {});

        return true;
    }

    ////

    #define TMP_MACRO(Char, o) \
        virtual bool writeValue(const Char& value) {return writeInt(value);}

    BUILTIN_INT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Floats.
    //
    //----------------------------------------------------------------

public:

    template <typename Float>
    sysinline bool writeFloat(const Float& value)
    {
        auto writer = [&] (auto* ptr, auto size)
            {writeBuf(ptr, size);};

        return ::writeFloat<Char>(value, writer, {});
    }

    ////

    #define TMP_MACRO(Char, o) \
        virtual bool writeValue(const Char& value) {return writeFloat(value);}

    BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    StringStorage& vec;

};

//================================================================
//
// ReadStreamImpl
//
//================================================================

class ReadStreamImpl : public CfgReadStream
{

public:

    sysinline ReadStreamImpl(const StringRef& str)
    {
        this->ptr = str.ptr;
        this->end = str.ptr + str.size;
    }

    //----------------------------------------------------------------
    //
    // skipSpaces
    //
    //----------------------------------------------------------------

public:

    virtual void skipSpaces()
    {
        skipSpaceTab(ptr, end);
    }

    //----------------------------------------------------------------
    //
    // skipText
    //
    //----------------------------------------------------------------

public:

    virtual bool skipText(const StringRef& text)
    {
        return ::skipText(ptr, end, text);
    }

    //----------------------------------------------------------------
    //
    // bool
    //
    //----------------------------------------------------------------

    virtual bool readValue(bool& result)
    {
        int tmp = 0;
        ensure(readInt(ptr, end, tmp));
        result = (tmp != 0);
        return true;
    }

    //----------------------------------------------------------------
    //
    // Signed int.
    //
    //----------------------------------------------------------------

public:

    #define TMP_MACRO(Type) \
        virtual bool readValue(Type& result) {return readInt(ptr, end, result);}

    TMP_MACRO(signed char)
    TMP_MACRO(signed short)
    TMP_MACRO(signed int)
    TMP_MACRO(signed long)
    TMP_MACRO(signed long long)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Unsigned int.
    //
    //----------------------------------------------------------------

public:

    #define TMP_MACRO(Type) \
        virtual bool readValue(Type& result) {return readUint(ptr, end, result);}

    TMP_MACRO(unsigned char)
    TMP_MACRO(unsigned short)
    TMP_MACRO(unsigned int)
    TMP_MACRO(unsigned long)
    TMP_MACRO(unsigned long long)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Builtin floats.
    //
    //----------------------------------------------------------------

    virtual bool readValue(float32& result)
    {
        float64 tmp{};
        ensure(readFloatApprox(ptr, end, tmp));
        result = float32(tmp);
        ensure(def(result));
        return true;
    }

    virtual bool readValue(float64& result)
    {
        ensure(readFloatApprox(ptr, end, result));
        ensure(def(result));
        return true;
    }

    //----------------------------------------------------------------
    //
    // readAll
    //
    //----------------------------------------------------------------

public:

    virtual bool readAll(CfgOutputString& result)
    {
        ensure(result.addBuf(ptr, end - ptr));
        ptr = end;
        return true;
    }

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    const Char* ptr = nullptr;
    const Char* end = nullptr;

};

//================================================================
//
// getVarName
//
//================================================================

void getVarName(const CfgSerializeVariable& var, StringStorage& name)
{
    name.clear();

    auto appendToName = cfgOutputString | [&] (auto ptr, auto size)
    {
        name.append(ptr, size, true);
        return true;
    };

    if_not (var.getName(appendToName))
        name.clear();
}

//================================================================
//
// saveVarToNode
//
//================================================================

bool saveVarToNode
(
    const CfgSerializeVariable& var,
    CfgTree& varNode,
    bool updateSyncedFlag,
    CfgTemporary& temp
)
{
    WriteStreamImpl value{temp.value};
    WriteStreamImpl comment{temp.comment};
    WriteStreamImpl blockComment{temp.blockComment};

    ////

    value.clear();

    ensure(var.getTextValue(value));

    ////

    comment.clear();

    ensure(var.getTextComment(comment));

    ////

    blockComment.clear();

    ensure(var.getBlockComment(blockComment));

    ////

    cfgTree::SetDataArgs args;
    args.value = value.str();
    args.comment = comment.str();
    args.blockComment = blockComment.str();

    varNode.setData(args);

    ////

    if (updateSyncedFlag)
        var.setSynced(true);

    return true;
}

//================================================================
//
// saveAllVarsToTree
//
//================================================================

stdbool saveAllVarsToTree(const SaveVarsToTreeArgs& args, stdPars(Kit))
{
    stdExceptBegin;

    using namespace cfgTree;

    //----------------------------------------------------------------
    //
    // Scope visitor.
    //
    //----------------------------------------------------------------

    auto* spaceNode = &args.cfgTree;

    ////

    struct ScopeContext
    {
        Node* savedNode;
    };

    auto enter = [&] (auto& context, auto& name)
    {
        auto& the = context.template recast<ScopeContext>();
        the.savedNode = spaceNode;
        spaceNode = spaceNode->findOrCreateChild(NameRef{name});
    };

    auto leave = [&] (auto& context)
    {
        auto& the = context.template recast<ScopeContext>();
        spaceNode = the.savedNode;
        the.savedNode = nullptr;
    };

    auto scopeVisitor = cfgScopeVisitor(enter, leave);

    //----------------------------------------------------------------
    //
    // Variable visitor.
    //
    //----------------------------------------------------------------

    auto visitVar = cfgVisitVar | [&] (auto& var)
    {
        auto& name = args.temp.name;
        getVarName(var, name);
        ensurev(CHECK(name.size() != 0));

        ////

        auto* varNode = spaceNode->findOrCreateChild(NameRef{name});

        ensurev(CHECK(saveVarToNode(var, *varNode, args.updateSyncedFlag, args.temp)));
    };

    //----------------------------------------------------------------
    //
    // Visit.
    //
    //----------------------------------------------------------------

    args.serialization({visitVar, CfgVisitSignalNull{}, scopeVisitor});

    ////

    stdExceptEnd;
}

//================================================================
//
// saveUnsyncedVarsToTree
//
// Uses different strategy:
//
// For each variable of the small set of changed variables,
// create the full path from the root of the tree.
//
//================================================================

stdbool saveUnsyncedVarsToTree(const SaveVarsToTreeArgs& args, stdPars(Kit))
{
    stdExceptBegin;

    using namespace cfgTree;

    //----------------------------------------------------------------
    //
    // Scope visitor.
    //
    //----------------------------------------------------------------

    struct ScopeContext
    {
        StringRef name;
        ScopeContext* next;
        ScopeContext* savedScope;
    };

    ////

    ScopeContext preScope{STR("Pre-Scope"), nullptr, nullptr};
    auto* currentScope = &preScope;

    ////

    auto enter = [&] (auto& context, auto& name)
    {
        auto& the = context.template recast<ScopeContext>();

        the.name = name;
        the.savedScope = currentScope;

        currentScope->next = &the;
        currentScope = &the;
    };

    auto leave = [&] (auto& context)
    {
        auto& the = context.template recast<ScopeContext>();

        currentScope = the.savedScope;
    };

    auto scopeVisitor = cfgScopeVisitor(enter, leave);

    //----------------------------------------------------------------
    //
    // Variable visitor.
    //
    //----------------------------------------------------------------

    auto visitVar = cfgVisitVar | [&] (auto& var)
    {
        ensurev(!var.synced());

        ////

        auto& debugStr = args.temp.debugStr;
        debugStr.clear();

        //
        // Create/enter the full path in the tree.
        //

        int level = 0;

        auto* spaceNode = &args.cfgTree;

        for (auto* ptr = &preScope; ptr != currentScope; )
        {
            ptr = ptr->next;

            ++level;
            spaceNode = spaceNode->findOrCreateChild(NameRef{ptr->name});

            if (args.debugPrint)
                debugStr << STR("/") << ptr->name;
        }

        //
        // Create var node.
        //

        auto& name = args.temp.name;

        getVarName(var, name);

        ensurev(CHECK(name.size() != 0));

        auto* varNode = spaceNode->findOrCreateChild(NameRef{name});

        //
        // Update var node.
        //

        ensurev(CHECK(saveVarToNode(var, *varNode, args.updateSyncedFlag, args.temp)));

        ////

        if (args.debugPrint)
        {
            debugStr << STR("/") << name;
            printMsg(kit.msgLog, STR("Updating %=%"), debugStr.str(), args.temp.value.str());
        }
    };

    //----------------------------------------------------------------
    //
    // Visit.
    //
    //----------------------------------------------------------------

    args.serialization({visitVar, CfgVisitSignalNull{}, scopeVisitor});

    ////

    stdExceptEnd;
}

//================================================================
//
// saveVarsToTree
//
//================================================================

stdbool saveVarsToTree(const SaveVarsToTreeArgs& args, stdPars(Kit))
{
    return args.saveOnlyUnsyncedVars ?
        saveUnsyncedVarsToTree(args, stdPass) :
        saveAllVarsToTree(args, stdPass);
}

//================================================================
//
// loadVarsFromTree
//
//================================================================

stdbool loadVarsFromTree(const LoadVarsFromTreeArgs& args, stdPars(Kit))
{
    stdExceptBegin;

    using namespace cfgTree;

    //----------------------------------------------------------------
    //
    // Scope visitor.
    //
    //----------------------------------------------------------------

    auto* spaceNode = &args.cfgTree; // May be NULL

    ////

    struct ScopeContext
    {
        Node* savedNode;
    };

    auto enter = [&] (auto& context, auto& name)
    {
        auto& the = context.template recast<ScopeContext>();

        the.savedNode = spaceNode;

        if (spaceNode)
            spaceNode = spaceNode->findChild(NameRef{name});;
    };

    auto leave = [&] (auto& context)
    {
        auto& the = context.template recast<ScopeContext>();
        spaceNode = the.savedNode;
        the.savedNode = nullptr;
    };

    auto scopeVisitor = cfgScopeVisitor(enter, leave);

    //----------------------------------------------------------------
    //
    // Variable visitor.
    //
    //----------------------------------------------------------------

    auto visitVar = cfgVisitVar | [&] (auto& var)
    {

        ensurev(spaceNode); // No sub-tree.

        ////

        if (args.loadOnlyUnsyncedVars)
        {
            ensurev(!var.synced());
        }

        //----------------------------------------------------------------
        //
        // Get var name.
        //
        //----------------------------------------------------------------

        auto& name = args.temp.name;

        getVarName(var, name);

        ensurev(name.size() != 0);

        //----------------------------------------------------------------
        //
        // Find var node.
        //
        //----------------------------------------------------------------

        auto* varNode = spaceNode->findChild(NameRef{name});

        ensurev(varNode);

        //----------------------------------------------------------------
        //
        // Update value.
        //
        // If the variable value successfully loaded, clear the changed flag.
        //
        //----------------------------------------------------------------

        ensurev(varNode->hasData());

        auto value = varNode->getData().value;

        auto readStream = ReadStreamImpl{value};

        ensurev(var.setTextValue(readStream));

        if (args.updateSyncedFlag)
            var.setSynced(true);

    };

    //----------------------------------------------------------------
    //
    // Visit.
    //
    //----------------------------------------------------------------

    args.serialization({visitVar, CfgVisitSignalNull{}, scopeVisitor});

    ////

    stdExceptEnd;
}

//----------------------------------------------------------------

}
