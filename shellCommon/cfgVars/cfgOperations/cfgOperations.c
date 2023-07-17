#include "cfgOperations.h"

#include "binaryFile/binaryFileImpl.h"
#include "compileTools/blockExceptionsSilent.h"
#include "cfgVars/types/stringStorage.h"
#include "cfgVars/cfgSerializeImpl/cfgSerializeImpl.h"
#include "cfgVars/cfgTree/cfgTree.h"
#include "data/array.h"
#include "errorLog/convertExceptions.h"
#include "errorLog/debugBreak.h"
#include "errorLog/errorLog.h"
#include "interfaces/fileTools.h"
#include "numbers/int/intType.h"
#include "parseTools/parseTools.h"
#include "podVector/podVector.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"

namespace cfgOperations {

using namespace std;
using namespace cfgSerializeImpl;

//================================================================
//
// InputStream
//
//================================================================

class InputStream
{

public:

    sysinline InputStream(const StringRef& str)
    {
        this->ptr = str.ptr;
        this->end = str.ptr + str.size;
    }

    sysinline bool getLine(StringRef& s, bool& eos)
    {
        s = {};
        eos = (ptr == end);

        const Char* resPtr{};
        const Char* resEnd{};
        ensure(getNextLine(ptr, end, resPtr, resEnd));

        s = {resPtr, size_t(resEnd - resPtr)};
        return true;
    }

private:

    const Char* ptr = nullptr;
    const Char* end = nullptr;

};

//================================================================
//
// endl
//
//================================================================

struct Endline {};
constexpr Endline endl;

//================================================================
//
// RepeatedChar
//
//================================================================

struct RepeatedChar
{
    Char value;
    size_t count;
};

//================================================================
//
// OutputStream
//
//================================================================

class OutputStream
{

public:

    sysinline OutputStream(size_t reservedSize)
        {vec.reserve(reservedSize);}

public:

    sysinline const Char* ptr() const
        {return vec.data();}

    sysinline auto size() const
        {return vec.size();}

public:

    sysinline void write(const Char* ptr, size_t size)
        {vec.append(ptr, size, true);}

public:

    sysinline auto& operator <<(const StringRef& str)
    {
        write(str.ptr, str.size);
        return *this;
    }

    sysinline auto& operator <<(const Endline& value)
    {
        vec.push_back('\n');
        return *this;
    }

    sysinline auto& operator <<(const RepeatedChar& that)
    {
        vec.append_value(that.value, that.count, true);
        return *this;
    }

private:

    PodVector<Char> vec;

};

//================================================================
//
// getTemporaryFilenameCstr
//
//================================================================

inline void getTemporaryFilenameCstr(const StringRef& filename, StringStorage& result)
{
    result = filename;
    result.append(STR(".temporary"));
    result.push_back(0);
}

//================================================================
//
// parseTextString
//
// Can throw exceptions.
// If parsing is not successful, does not move the pointer.
//
//================================================================

bool parseTextString(const Char*& ptr, const Char* end, StringStorage& dst)
{
    dst.clear();

    auto writer = [&] (auto ptr, auto size)
    {
        dst.append(ptr, size, true);
    };

    ensure(decodeJsonStr(ptr, end, writer));

    return true;
}

//================================================================
//
// skipJsonNumber
//
//================================================================

sysinline bool skipJsonNumber(const Char*& ptr, const Char* end)
{
    return skipFloat(ptr, end, false);
}

//================================================================
//
// skipJsonFloatSeq
//
// If parsing is not successful, does not move the pointer.
//
//================================================================

bool skipJsonFloatSeq(const Char*& ptr, const Char* end)
{
    auto s = ptr;

    ensure(skipJsonNumber(s, end));
    auto nonSpacePoint = s;
    skipAnySpace(s, end);

    while (skipTextThenAnySpace(s, end, STR(",")))
    {
        ensure(skipJsonNumber(s, end));
        nonSpacePoint = s;
        skipAnySpace(s, end);
    }

    ptr = nonSpacePoint;
    return true;
}

//================================================================
//
// parseValueString
//
//================================================================

bool parseValueString(const Char*& ptr, const Char* end, StringStorage& result)
{
    if (parseTextString(ptr, end, result))
        return true;

    ////

    auto numStart = ptr;

    if (skipJsonNumber(ptr, end))
    {
        result.assign(numStart, ptr - numStart, false);
        return true;
    }

    ////

    auto s = ptr;
    ensure(skipTextThenAnySpace(s, end, STR("[")));

    auto seqStart = s;
    ensure(skipJsonFloatSeq(s, end));
    auto seqEnd = s;

    skipAnySpace(s, end);
    ensure(skipText(s, end, STR("]")));

    result.assign(seqStart, seqEnd - seqStart, false);

    ptr = s;
    return true;
}

//================================================================
//
// saveString
//
// Can throw exceptions.
//
//================================================================

void saveString(const StringRef& str, OutputStream& stream, bool valueString)
{
    auto ptr = str.ptr;
    auto end = str.ptr + str.size;

    ////

    {
        auto s = ptr;

        if (skipJsonNumber(s, end) && s == end)
        {
            stream << str;
            return;
        }
    }

    ////

    {
        auto s = ptr;

        if (skipJsonFloatSeq(s, end) && s == end)
        {
            stream << STR("[") << str << STR("]");
            return;
        }
    }

    ////

    auto writer = [&] (auto* ptr, auto size)
    {
        stream.write(ptr, size);
    };

    stream << STR("\"");
    encodeJsonStr(ptr, end, writer);
    stream << STR("\"");
}

//================================================================
//
// loadFromStream
//
// Updates the memory from stream. May throw exceptions.
//
//================================================================

stdbool loadFromStream(InputStream& stream, CfgTree& memory, const StringRef& filename, bool trackDataChange, CfgTemporary& temp, stdPars(Kit))
{
    using namespace cfgTree;

    //----------------------------------------------------------------
    //
    // Local memory.
    //
    //----------------------------------------------------------------

    PodVector<Node*> treeStack;
    treeStack.reserve(32);

    Node* scopeNode = &memory;

    ////

    auto& name = temp.name;
    auto& varValue = temp.value;
    auto& blockComment = temp.blockComment;

    ////

    size_t lineIndex = 0;

    while (true)
    {

        //----------------------------------------------------------------
        //
        // Read line.
        //
        //----------------------------------------------------------------

        ++lineIndex;

        bool eos{};

        StringRef textLine;

        if_not (stream.getLine(textLine, eos))
        {
            REQUIRE(eos);
            break;
        }

        ////

        REMEMBER_CLEANUP_EX(resetBlockComment, blockComment.clear());

        //----------------------------------------------------------------
        //
        // Parse text line.
        //
        //----------------------------------------------------------------

        breakBlock(parseOneLine)
        {

            //----------------------------------------------------------------
            //
            // Skip leading blanks.
            // Skip empty lines.
            //
            //----------------------------------------------------------------

            const auto* ptr = textLine.ptr;
            const auto* end = textLine.ptr + textLine.size;

            ////

            skipSpaceTab(ptr, end);

            if (ptr == end)
                breakTrue;

            //----------------------------------------------------------------
            //
            // Block comment.
            //
            //----------------------------------------------------------------

            breakBlock(blockCommentSyntax)
            {
                auto* p = ptr;

                // comment
                breakRequire(skipTextThenAnySpace(p, end, STR("//")));

                if (blockComment.size())
                    blockComment.push_back('\n');

                blockComment.append(p, end - p, true);
            }

            if (blockCommentSyntax)
            {
                resetBlockComment.cancel();
                breakTrue;
            }

            //----------------------------------------------------------------
            //
            // Variable syntax.
            //
            //----------------------------------------------------------------

            breakBlock(varSyntax)
            {
                auto* p = ptr;

                breakRequire(parseTextString(p, end, name));
                skipAnySpace(p, end);

                breakRequire(skipTextThenAnySpace(p, end, STR(":")));

                breakRequire(parseValueString(p, end, varValue));
                skipAnySpace(p, end);

                skipTextThenAnySpace(p, end, STR(",")); // Optional

                StringRef valueComment;

                breakBlock(getValComment)
                {
                    breakRequire(skipTextThenAnySpace(p, end, STR("//")));
                    valueComment = {p, size_t(end - p)};
                    p = end;
                }

                breakRequire(p == end);

                //
                // successfully recognized
                //

                auto varNode = scopeNode->findOrCreateChild(NameRef{name});

                SetDataArgs args;
                args.value = varValue;
                args.comment = valueComment;
                args.blockComment = blockComment;

                if (trackDataChange)
                    varNode->setDataEx(args);
                else
                    varNode->setData(args);
            }

            if (varSyntax)
                breakTrue;

            //----------------------------------------------------------------
            //
            // Space end:
            // }
            //
            //----------------------------------------------------------------

            breakBlock(spaceCloseSyntax)
            {
                auto* p = ptr;
                breakRequire(skipTextThenAnySpace(p, end, STR("}")));
                skipTextThenAnySpace(p, end, STR(",")); // Optional.
                breakRequire(p == end);

                ////

                if (treeStack.size())
                {
                    scopeNode = treeStack.back();
                    treeStack.pop_back();
                }
            }

            if (spaceCloseSyntax)
                breakTrue;

            //----------------------------------------------------------------
            //
            // Space begin.
            //
            //----------------------------------------------------------------

            breakBlock(spaceOpenNamespace)
            {
                auto* p = ptr;

                breakRequire(parseTextString(p, end, name));
                skipAnySpace(p, end);

                breakRequire(skipTextThenAnySpace(p, end, STR(":")));

                breakRequire(p == end);

                ////

                auto newScope = scopeNode->findOrCreateChild(NameRef{name});
                treeStack.push_back(scopeNode);
                scopeNode = newScope;
            }

            if (spaceOpenNamespace)
                breakTrue;

            //----------------------------------------------------------------
            //
            // Space open brace.
            //
            //----------------------------------------------------------------

            breakBlock(spaceOpenBrace)
            {
                auto* p = ptr;
                breakRequire(skipTextThenAnySpace(p, end, STR("{")));

                breakRequire(p == end);
            }

            if (spaceOpenBrace)
                breakTrue;

            ////

            breakFalse;
        }

        //----------------------------------------------------------------
        //
        // Cannot parse line?
        //
        //----------------------------------------------------------------

        if_not (parseOneLine)
            printMsg(kit.msgLog, STR("Config: Cannot parse line %(%)"), filename, lineIndex, msgWarn);

    }

    returnTrue;
}

//================================================================
//
// saveVar
//
// Can throw exceptions
//
//================================================================

void saveVar
(
    const RepeatedChar& indent,
    const StringRef& name,
    const StringRef& value,
    const StringRef& valueComment,
    const StringRef& blockComment,
    OutputStream& stream,
    bool comma
)
{
    //----------------------------------------------------------------
    //
    // Block comment.
    //
    //----------------------------------------------------------------

    auto& comment = blockComment;

    if (comment.size)
    {
        auto ptr = comment.ptr;
        auto end = comment.ptr + comment.size;

        for (;;)
        {
            auto start = ptr;

            while (ptr != end && *ptr != '\n')
                ++ptr;

            StringRef line{start, size_t(ptr - start)};

            if (ptr != end)
                ++ptr;

            stream << indent << (line.size ? STR("// ") : STR("//")) << line << endl;

            if (ptr == end)
                break;
        }
    }

    //----------------------------------------------------------------
    //
    // Value.
    //
    //----------------------------------------------------------------

    stream << indent;

    saveString(name, stream, false);

    stream << STR(": ");

    saveString(value, stream, true);

    //----------------------------------------------------------------
    //
    // Value comment.
    //
    //----------------------------------------------------------------

    if (comma)
        stream << STR(",");

    if (valueComment.size)
        stream << STR(" // ") << valueComment;

    ////

    stream << endl;
}

//================================================================
//
// Settings.
//
//================================================================

constexpr int indentWidth = 4;
constexpr int separatorWidth = 64;

//================================================================
//
// LevelVars
//
//================================================================

struct LevelVars
{
    PodVector<CfgTree*> index;

    void dealloc()
        {index.dealloc();}

    void reserve(size_t n)
        {index.reserve(n);}
};

//================================================================
//
// SaveTreeArgs
//
//================================================================

struct SaveTreeArgs
{
    OutputStream& stream;
    Array<LevelVars> levelVars;
};

//================================================================
//
// saveTree
//
// Can throw exceptions.
//
//================================================================

void saveTree(const SaveTreeArgs& args, size_t level, CfgTree& root)
{
    auto& stream = args.stream;
    auto& levelVars = args.levelVars;

    auto indent = RepeatedChar{' ', level * indentWidth};

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    enum class State {None, Line, Big};
    State prevState = State::None;

    //----------------------------------------------------------------
    //
    // Open brace.
    //
    //----------------------------------------------------------------

    if (level == 0)
    {
        stream << STR("{") << endl;
        prevState = State::Big;
    }

    //----------------------------------------------------------------
    //
    // Save subtree function.
    //
    //----------------------------------------------------------------

    auto saveSubtree = [&] (auto& node, bool last)
    {
        if (level == 0)
        {
            if (prevState != State::None)
                stream << endl;

            auto blockLine = RepeatedChar{'=', separatorWidth};

            stream << indent << STR("//") << blockLine << endl;
            stream << indent << STR("//") << endl;
            stream << indent << STR("// ") << node.getName() << endl;
            stream << indent << STR("//") << endl;
            stream << indent << STR("//") << blockLine << endl;

            prevState = State::Big;
        }

        ////

        if (prevState != State::None)
            stream << endl;

        stream << indent;
        saveString(node.getName(), stream, false);
        stream << STR(":") << endl;

        stream << indent << STR("{") << endl;

        saveTree(args, level + 1, node);

        stream << indent << (!last ? STR("},") : STR("}")) << endl;

        prevState = State::Big;
    };

    //----------------------------------------------------------------
    //
    // Save leaf function.
    //
    //----------------------------------------------------------------

    auto saveLeaf = [&] (auto& node, bool last)
    {
        auto data = node.getData();

        bool big = data.blockComment.size != 0;

        if (prevState == State::Big || (big && prevState != State::None))
            stream << endl;

        saveVar
        (
            indent,
            node.getName(),
            data.value,
            data.comment,
            data.blockComment,
            stream,
            !last
        );

        prevState = big ? State::Big : State::Line;
    };

    //----------------------------------------------------------------
    //
    // Gather tasks.
    //
    //----------------------------------------------------------------

    LevelVars levelVarsLocal;

    auto& lv = (level < size_t(levelVars.size())) ? levelVars[Space(level)] : levelVarsLocal;

    ////

    lv.index.clear();

    auto handler = cfgTree::NodeHandler::O | [&] (auto& node)
    {
        if (node.hasChildren() || node.hasData())
            lv.index.push_back(&node);
    };

    root.forAllChildren(handler);

    ////

    auto nodeCount = lv.index.size();

    for_count (i, nodeCount)
    {
        auto& node = *lv.index[i];

        bool last = (i == nodeCount - 1);

        if (node.hasChildren())
            saveSubtree(node, last && !node.hasData());

        if (node.hasData())
            saveLeaf(node, last);
    }

    //----------------------------------------------------------------
    //
    // Global brace.
    //
    //----------------------------------------------------------------

    if (level == 0)
    {
        if (prevState != State::None)
            stream << endl;

        if (prevState == State::Big)
            stream << STR("//") << RepeatedChar{'-', separatorWidth} << endl << endl;

        stream << STR("}") << endl;
    }

}

//================================================================
//
// CfgOperationsImpl
//
//================================================================

class CfgOperationsImpl : public CfgOperations
{

    //----------------------------------------------------------------
    //
    // Dealloc.
    //
    //----------------------------------------------------------------

    void dealloc()
    {
        temp.dealloc();

        for_count (i, maxStaticLevels)
            levelVarsArr[i].dealloc();
    }

    //----------------------------------------------------------------
    //
    // File I/O.
    //
    //----------------------------------------------------------------

    stdbool loadFromFile(CfgTree& memory, const Char* filename, bool trackDataChange, stdPars(Kit));
    stdbool saveToFile(CfgTree& memory, const Char* filename, stdPars(Kit));

    //----------------------------------------------------------------
    //
    // String I/O.
    //
    //----------------------------------------------------------------

    stdbool loadFromString(CfgTree& memory, const StringRef& str, stdPars(Kit))
    {
        stdExceptBegin;

        temp.reserve();

        InputStream stream{str};
        require(loadFromStream(stream, memory, STR("memoryFile"), false, temp, stdPass));

        stdExceptEnd;
    }

    stdbool saveToString(CfgTree& memory, StringReceiver& receiver, stdPars(Kit))
    {
        stdExceptBegin;

        OutputStream stream{wholeConfigReserve};
        require(saveToStream(memory, stream, stdPass));
        require(receiver(charArray(stream.ptr(), stream.size()), stdPass));

        stdExceptEnd;
    }

    //----------------------------------------------------------------
    //
    // Serialization.
    //
    //----------------------------------------------------------------

    virtual stdbool saveVars(CfgTree& memory, CfgSerialization& serialization, const SaveVarsOptions& options, stdPars(Kit))
    {
        stdExceptBegin;

        temp.reserve();

        SaveVarsToTreeArgs args{serialization, memory, temp, options.saveOnlyUnsyncedVars, options.updateSyncedFlag, false};
        require(saveVarsToTree(args, stdPass));

        stdExceptEnd;
    }

    ////

    virtual stdbool loadVars(CfgTree& memory, CfgSerialization& serialization, const LoadVarsOptions& options, stdPars(Kit))
    {
        stdExceptBegin;

        temp.reserve();

        LoadVarsFromTreeArgs args{serialization, memory, temp, options.loadOnlyUnsyncedVars, options.updateSyncedFlag};
        require(loadVarsFromTree(args, stdPass));

        stdExceptEnd;
    }

    //----------------------------------------------------------------
    //
    // saveToStream
    //
    // May throw exceptions.
    //
    //----------------------------------------------------------------

private:

    stdbool saveToStream(CfgTree& memory, OutputStream& stream, stdPars(Kit))
    {
        auto levelVars = makeArray(levelVarsArr, maxStaticLevels);

        constexpr int commonLevels = 4;
        COMPILE_ASSERT(commonLevels <= maxStaticLevels);

        for_count (i, commonLevels)
            levelVars[i].reserve(16);

        ////

        auto args = SaveTreeArgs{stream, levelVars};

        saveTree(args, 0, memory);

        ////

        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Temporary variables.
    //
    //----------------------------------------------------------------

private:

    CfgTemporary temp;

    ////

    static constexpr int maxStaticLevels = 16;
    LevelVars levelVarsArr[maxStaticLevels];

};

////

UniquePtr<CfgOperations> CfgOperations::create()
    {return makeUnique<CfgOperationsImpl>();}

//================================================================
//
// CfgOperationsImpl::loadFromFile
//
//================================================================

stdbool CfgOperationsImpl::loadFromFile(CfgTree& memory, const Char* filename, bool trackDataChange, stdPars(Kit))
{
    stdExceptBegin;

    auto filenameStr = charArrayFromPtr(filename);
    REQUIRE(filenameStr.size != 0);

    temp.reserve();

    //----------------------------------------------------------------
    //
    // Finish processes that could be interrupted when writing:
    //
    // Temporary file existense means the saving was not successfully completed,
    // so delete the file.
    //
    //----------------------------------------------------------------

    {
        auto& tmpFilename = temp.tmpFilename;
        getTemporaryFilenameCstr(filenameStr, tmpFilename);
        fileTools::deleteFile(tmpFilename.data());
    }

    //----------------------------------------------------------------
    //
    // Read from file.
    //
    //----------------------------------------------------------------

    BinaryFileImpl file;
    require(file.open(filenameStr, false, false, stdPass));

    auto rawSize = file.getSize();

    size_t size{};
    REQUIRE(convertExact(rawSize, size));
    size /= sizeof(Char);

    StringStorage content;
    content.resize(size, false);

    require(file.read(content.data(), size * sizeof(Char), stdPass));

    ////

    InputStream stream{content};
    require(loadFromStream(stream, memory, filenameStr, trackDataChange, temp, stdPass));

    ////

    stdExceptEnd;
}

//================================================================
//
// CfgOperationsImpl::saveToFile
//
//================================================================

stdbool CfgOperationsImpl::saveToFile(CfgTree& memory, const Char* filename, stdPars(Kit))
{
    stdExceptBegin;

    temp.reserve();

    ////

    auto filenameStr = charArrayFromPtr(filename);
    REQUIRE(filenameStr.size != 0);

    ////

    auto& tmpFilename = temp.tmpFilename;
    getTemporaryFilenameCstr(filenameStr, tmpFilename);

    auto tmpFilenameCstr = tmpFilename.data();

    auto tmpFilenameStr = tmpFilename.str();
    REQUIRE(tmpFilenameStr.size >= 1);
    tmpFilenameStr.size -= 1;

    //----------------------------------------------------------------
    //
    // Save to a temporary file
    //
    //----------------------------------------------------------------

    REMEMBER_CLEANUP_EX(tmpEraser, fileTools::deleteFile(tmpFilenameCstr));

    ////

    {
        OutputStream stream{wholeConfigReserve};

        require(saveToStream(memory, stream, stdPass));

        BinaryFileImpl file;
        require(file.open(tmpFilenameStr, true, true, stdPass));
        require(file.truncate(stdPass));

        require(file.write(stream.ptr(), stream.size() * sizeof(*stream.ptr()), stdPass));
    }

    //----------------------------------------------------------------
    //
    // Rename successfully updated file to the destination file.
    //
    //----------------------------------------------------------------

    REQUIRE(fileTools::renameFile(tmpFilenameCstr, filename));
    tmpEraser.cancel();

    ////

    stdExceptEnd;
}

//----------------------------------------------------------------

}
