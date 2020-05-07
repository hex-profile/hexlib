#include "cfgFileEnv.h"

#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#include <string.h>

#include "errorLog/errorLog.h"

namespace cfgVarsImpl {

using namespace std;

//================================================================
//
// TmpFileEraserThunk
//
//----------------------------------------------------------------
//
// Erases a temporary file in destructor
//
//================================================================

class TmpFileEraserThunk
{

public:

    using String = basic_string<CharType>;

public:

    inline TmpFileEraserThunk(const String& name, FileTools& fileTools)
        :
        theName(name),
        fileTools(fileTools)
    {
    }

    inline ~TmpFileEraserThunk()
    {
        if (active)
            fileTools.deleteFile(theName.c_str());
    }

    inline void cancel() {active = false;}

private:

    bool active = true;
    String theName;
    FileTools& fileTools;

};

//================================================================
//
// Record
//
//================================================================

struct Record
{
    String blockComment;
    String value;
    String valueComment;
};

//================================================================
//
// Memory
//
//================================================================

using Memory = map<NameContainer, Record>;

//================================================================
//
// temporaryFilename
//
//================================================================

static inline String getTemporaryFilename(const String& filename)
    {return filename + CT(".temporary");}

//================================================================
//
// Syntax tools
//
//================================================================

inline bool isSpace(CharType c)
{
    return
        c == ' ' ||
        c == '\t' ||
        c == '\v' ||
        c == '\r' ||
        c == '\n' ||
        c == '\f';
}

//================================================================
//
// parseString
//
// Can throw exceptions
//
//================================================================

static const CharType quoteBeg = '\'';
static const CharType quoteEnd = '\'';

//----------------------------------------------------------------

static bool parseString(const CharType*& p, String& result)
{
    // then comes string in quotes
    ensure(*p == quoteBeg); ++p;

    size_t n = strlen(p);
    vector<CharType> tmp(n); // for max number
    CharType* q = n ? &tmp[0] : 0;

    bool literally = false;

    for (;;)
    {
        if (*p == 0)
            break;

        // if it goes after the prefix `\`
        if (literally)
            {*q++ = *p++; literally = false; continue;}

        if (*p == quoteBeg || *p == quoteEnd)
            break;

        if (*p == '\\')
            {p++; literally = true; continue;}

        *q++ = *p++;
    }

    // String should end in appropriate way
    ensure(!literally && *p == quoteEnd);
    p++;

    // Transfer the string
    result.assign(&tmp[0], q - &tmp[0]);

    return true;
}

//================================================================
//
// saveString
//
// Can throw exceptions
//
//================================================================

static void saveString(const String& str, basic_ostream<CharType>& stream)
{

    //
    // count the number of additional symbols \ and "
    //

    stream << quoteBeg;

    const CharType* valuePtr = str.c_str();
    String::size_type valueSize = str.size();

    String::size_type nExtraChars = 0;

    for_count (i, valueSize)
    {
        if (valuePtr[i] == quoteBeg || valuePtr[i] == quoteEnd || valuePtr[i] == '\\')
            ++nExtraChars;
    }

    //
    // reencode to array and save to stream
    //

    vector<CharType> encodedVector(valueSize + nExtraChars);
    CharType* encodedPtr = NULL;
    if (encodedVector.size() > 0)
        encodedPtr = &encodedVector[0];

    for_count (i, valueSize)
    {
        if (valuePtr[i] == quoteBeg || valuePtr[i] == quoteEnd || valuePtr[i] == '\\')
            *encodedPtr++ = '\\';

        *encodedPtr++ = valuePtr[i];
    }

    if (encodedVector.size() > 0)
        stream.write(&encodedVector[0], encodedVector.size());

    stream << quoteEnd;

}

//================================================================
//
// parseNameId
//
// ID recognition
//
//================================================================

using NameID = uint32;

static const NameID nameIdNull = 0;
static const size_t nameIdBits = 32;

COMPILE_ASSERT(nameIdBits % 4 == 0);
static const size_t nameIdHexDigs = nameIdBits / 4;

//----------------------------------------------------------------

static bool parseNameId(const CharType*& p, NameID& result)
{
    result = 0;

    ////

    for_count (I, nameIdHexDigs)
    {
        CharType c = *p;
        int32 digit = 0;

        if (c >= '0' && c <= '9')
            digit = c - '0';
        else if (c >= 'a' && c <= 'f')
            digit = 10 + c - 'a';
        else if (c >= 'A' && c <= 'F')
            digit = 10 + c - 'A';
        else
            return false;

        ++p;
        result = (result << 4) + digit;
    }

    return true;
}

//================================================================
//
// saveNameId
//
// (can throw exceptions)
//
//================================================================

static void saveNameId(const NameID& id, basic_ostream<CharType>& stream)
{
    CharType buffer[nameIdHexDigs];

    uint32 V = id;

    for_count (I, 8)
    {
        buffer[I] = "0123456789ABCDEF"[(V >> (32 - 4)) & 0x0F];
        V <<= 4;
    }

    stream.write(buffer, nameIdHexDigs);
}

//================================================================
//
// loadFile
//
// Reading file and parsing
//
//================================================================

static const CharType valueSeparator = '=';

//----------------------------------------------------------------

stdbool loadFile(const CharType* filename, Memory& memory, FileTools& fileTools, stdPars(ErrorLogKit))
{
    try
    {

        String temporaryFilename = getTemporaryFilename(filename);

        //----------------------------------------------------------------
        //
        // Finish processes that could be violently interrupted when writing:
        //
        // If there is a temporary file, this means it was not successfully written until the end, so delete it.
        //
        //----------------------------------------------------------------

        fileTools.deleteFile(temporaryFilename.c_str());

        //----------------------------------------------------------------
        //
        // reading from file to a new container
        //
        //----------------------------------------------------------------

        Memory newMemory;

        {

            basic_ifstream<CharType> stream(filename);
            require(!!stream);

            //
            //
            //

            // block comment
            String blockComment;

            // current namespace
            NameContainer spacePrefix;
            // spacePrefix.reserve(32);

            //
            //
            //

            while (true)
            {

                //----------------------------------------------------------------
                //
                // reading line
                //
                //----------------------------------------------------------------

                String s;
                getline(stream, s);

                if (!stream)
                {
                    REQUIRE(stream.eof());
                    break;
                }

                //----------------------------------------------------------------
                //
                // null-terminated string for recognition
                //
                //----------------------------------------------------------------

                breakBlock(parseOneLine)
                {

                    //
                    // ignore blanks in any case
                    //

                    const CharType* str = s.c_str();
                    breakRequire(*str != 0);
                    while (isSpace(*str)) ++str;

                    //----------------------------------------------------------------
                    //
                    // recognize syntax of leading comment
                    // // comment
                    //
                    //----------------------------------------------------------------

                    breakBlock(blockCommentSyntax)
                    {
                        const CharType* p = str;

                        // comment
                        breakRequire(*p == ';'); ++p;

                        // then it can have space in our airy style
                        if (*p == ' ') ++p;

                        // successfully recognized
                        String commentStr = p;

                        if_not (blockComment.empty())
                            blockComment += CT("\n");

                        blockComment += commentStr;
                    }

                    if (blockCommentSyntax)
                        breakTrue;

                    //----------------------------------------------------------------
                    //
                    // recognize the syntax of space end:
                    // }
                    //
                    //----------------------------------------------------------------

                    breakBlock(spaceCloseSyntax)
                    {
                        const CharType* p = str;
                        breakRequire(*p == '}');

                        ++p;
                        while (isSpace(*p)) ++p;
                        breakRequire(*p == 0);

                        //
                        // ok
                        //

                        blockComment.clear();

                        if (!spacePrefix.empty())
                            spacePrefix.pop_back();

                    }

                    if (spaceCloseSyntax)
                        breakTrue;

                    //----------------------------------------------------------------
                    //
                    // recognize the syntax of space begin
                    //
                    //----------------------------------------------------------------

                    breakBlock(spaceOpenSyntax)
                    {
                        const CharType* p = str;

                        // spaces
                        while (isSpace(*p)) ++p;

                        breakRequire(*p == 'n'); ++p;
                        breakRequire(*p == 'a'); ++p;
                        breakRequire(*p == 'm'); ++p;
                        breakRequire(*p == 'e'); ++p;
                        breakRequire(*p == 's'); ++p;
                        breakRequire(*p == 'p'); ++p;
                        breakRequire(*p == 'a'); ++p;
                        breakRequire(*p == 'c'); ++p;
                        breakRequire(*p == 'e'); ++p;

                        while (isSpace(*p)) ++p;

                        // description
                        String spaceDesc;
                        breakRequire(parseString(p, spaceDesc));
                        while (isSpace(*p)) ++p;

                        //
                        // successfully recognized
                        //

                        blockComment.clear();
                        spacePrefix.push_back(NamePart(spaceDesc));
                    }

                    if (spaceOpenSyntax)
                        breakTrue;

                    //----------------------------------------------------------------
                    //
                    // recognize the syntax of a variable
                    //
                    //----------------------------------------------------------------

                    breakBlock(varSyntax)
                    {
                        const CharType* p = str;

                        // spaces
                        while (isSpace(*p)) ++p;

                        // description
                        String nameDesc;
                        breakRequire(parseString(p, nameDesc));

                        // then " = "
                        while (isSpace(*p)) ++p;
                        breakRequire(*p == valueSeparator); ++p;
                        while (isSpace(*p)) ++p;

                        // string with value
                        String varValue;
                        breakRequire(parseString(p, varValue));

                        // space
                        while (isSpace(*p)) ++p;

                        // then comment or the end of line
                        String valueComment;

                        breakBlock(getValComment)
                        {
                            breakRequire(*p == ';'); ++p;
                            if (isSpace(*p)) ++p;

                            const CharType* end = p + strlen(p);
                            valueComment.assign(p, end);
                            p = end;
                        }

                        // then spaces
                        while (isSpace(*p)) ++p;

                        // and the end of line
                        breakRequire(*p == 0);

                        //
                        // successfully recognized
                        //

                        NameContainer varName = spacePrefix;
                        varName.push_back(NamePart(nameDesc));

                        Memory::iterator i = newMemory.find(varName);

                        if (i != newMemory.end())
                        {
                            i->second.value = varValue;
                            i->second.blockComment = blockComment;
                            i->second.valueComment = valueComment;
                        }
                        else
                        {
                            Record rec;
                            rec.value = varValue;
                            rec.blockComment = blockComment;
                            rec.valueComment = valueComment;

                            newMemory.insert(pair<NameContainer, Record>(varName, rec));
                        }

                        blockComment.clear();
                    }

                    breakFalse;

                }

                //
                // cannot parse line, reset block comment
                //

                if_not (parseOneLine)
                    blockComment.clear();

            }

        }

        //
        // if it was able to read until this place, update the container
        //

        memory.swap(newMemory);

    }
    catch (const exception&) {REQUIRE(false);}

    returnTrue;
}

//================================================================
//
// Tree decomposition support, for saving to a file.
//
//================================================================

inline bool operator <(const NamePart& A, const NamePart& B)
    {return A.desc < B.desc;}

struct Node;

using Index = vector<Node>::size_type;

using Childs = map<NamePart, Index>;

//================================================================
//
// Node
//
//================================================================

struct Node
{
    // Has meaning only if there is no subnodes.
    const Record* record = nullptr;

    // Enumeration of subnode indices.
    Childs childs;
};

//================================================================
//
// Tree
//
//================================================================

using Tree = vector<Node>;

//================================================================
//
// treeAdd
//
//================================================================

static void treeAdd(Tree& container, const NameContainer& name, const Record& record)
{

    ////

    if (container.size() == 0)
        container.resize(1);

    ////

    if_not (name.size() >= 1)
        return;

    ////

    Index curNode = 0;

    ////

    for_count (i, name.size() - 1)
    {
        const NamePart& curName = name[i];

        Childs::const_iterator f = container[curNode].childs.find(curName);

        if (f != container[curNode].childs.end())
        {
            curNode = f->second;
        }
        else
        {
            container.resize(container.size() + 1);

            // Here and later: if an exception occurs after push_back, but before/in process of insertion,
            // some amount of memory simply remains unused

            Index newNode = container.size() - 1;

            container[curNode].childs.insert(pair<NamePart, Index>(curName, newNode));
            curNode = newNode;
        }
    }

    ////

    const NamePart& lastName = name.back();

    Childs::const_iterator f = container[curNode].childs.find(lastName);

    if (f != container[curNode].childs.end())
    {
        container[f->second].record = &record;
    }
    else
    {
        container.resize(container.size() + 1);
        Index newNode = container.size() - 1;

        container[newNode].record = &record;

        container[curNode].childs.insert(pair<NamePart, Index>(lastName, newNode));
    }

}

//================================================================
//
// saveVar
//
// Can throw exceptions
//
//================================================================

static void saveVar
(
    const String& indent,
    const NamePart* prefix,
    const NamePart& name,
    const Record& r,
    basic_ofstream<CharType>& stream
)
{

    //
    // save block comment
    //

    const String& comment = r.blockComment;

    if (!comment.empty())
    {
        if (comment.find(CT("\n")) == String::npos)
            stream << indent << CT("; ") << comment << endl;
        else
        {
            String::size_type pos = 0;

            for (;;)
            {
                String::size_type p = comment.find(CT("\n"), pos);

                if (p == String::npos)
                    break;

                stream << indent << CT("; ") << comment.substr(pos, p - pos) << endl;
                pos = p + 1;
            }

            stream << indent << CT("; ") << comment.substr(pos) << endl;
        }
    }

    //
    //
    //

    stream << indent;

    //
    // value
    //

    saveString(name.desc, stream);

    stream << CT(" ") << valueSeparator << CT(" ");

    saveString(r.value, stream);

    //
    // comment
    //

    if (!r.valueComment.empty())
    {
        stream << CT(" ; ") << r.valueComment;
    }

    //stream << CT(" ; ");
    //saveNameId(name.id, stream);

    //
    //
    //

    stream << endl;

}

//================================================================
//
// DescComparator
//
//================================================================

struct DescComparator
{

    inline bool operator()(Childs::const_iterator A, Childs::const_iterator B)
    {
        return A->first.desc < B->first.desc;
    }

};

//================================================================
//
// COMPACT_COUNT
//
// The number of nodes to use compaction.
//
//================================================================

static const int COMPACT_COUNT = 2;
static const int indentWidth = 4;

//================================================================
//
// saveTree
//
// Can throw exceptions.
//
//================================================================

static void saveTree
(
    String& indent,
    const Tree& tree,
    const Node& root,
    basic_ofstream<CharType>& stream,
    bool outerLevel = true,
    const NamePart* prefix = 0
)
{

    //
    // Sort subnodes by description
    //

    using ChildsIndex = vector<Childs::const_iterator>;

    ChildsIndex nodesIndex;
    ChildsIndex subtreesIndex;

    for (Childs::const_iterator i = root.childs.begin(); i != root.childs.end(); ++i)
    {
        if (tree[i->second].childs.empty())
            nodesIndex.push_back(i);
        else
            subtreesIndex.push_back(i);
    }

    sort(nodesIndex.begin(), nodesIndex.end(), DescComparator());
    sort(subtreesIndex.begin(), subtreesIndex.end(), DescComparator());

    //
    // Save
    //

    int prevFlag = 0; // 0 nothing, 1 line, 2 big

    for (ChildsIndex::const_iterator k = subtreesIndex.begin(); k != subtreesIndex.end(); ++k)
    {
        Childs::const_iterator i = *k;

        const Node& node = tree[i->second];

        if (outerLevel)
        {
            if (prevFlag) stream << endl;
            stream << indent << CT(";") << String(67, '=') << endl;
            stream << indent << CT(";") << endl;
            stream << indent << CT("; ") << i->first.desc << endl;
            stream << indent << CT(";") << endl;
            stream << indent << CT(";") << String(67, '=') << endl;
            prevFlag = 2;
        }

        {
            if (prevFlag) stream << endl;
            stream << indent;
            stream << CT("namespace ");
            saveString(i->first.desc, stream);
            //stream << CT(" ; ");
            //saveNameId(i->first.id, stream);
            stream << endl;

            stream << indent << CT("{") << endl;

            indent.resize(indent.size() + indentWidth, ' ');
            saveTree(indent, tree, node, stream, false);
            indent.resize(indent.size() - indentWidth);

            stream << indent << CT("}") << endl;
            prevFlag = 2;
        }

    }

    //
    // Leaves
    //

    for (ChildsIndex::const_iterator k = nodesIndex.begin(); k != nodesIndex.end(); ++k)
    {
        Childs::const_iterator i = *k;

        const Node& node = tree[i->second];

        bool big = !node.record->blockComment.empty();

        if (prevFlag == 2 || big)
            {stream << endl;}

        saveVar(indent, prefix, i->first, *node.record, stream);

        prevFlag = big ? 2 : 1;
    }

}

//================================================================
//
// saveFile
//
//================================================================

stdbool saveFile(const Memory& memory, const CharType* filename, FileTools& fileTools, stdPars(ErrorLogKit))
{
    try
    {

        REQUIRE(filename != 0 && *filename != 0);

        String temporaryFilename = getTemporaryFilename(filename);

        //----------------------------------------------------------------
        //
        // Decompose onto a tree
        //
        //----------------------------------------------------------------

        Tree tree;

        for (Memory::const_iterator i = memory.begin(); i != memory.end(); ++i)
            treeAdd(tree, i->first, i->second);

        //----------------------------------------------------------------
        //
        // Save to a temporary file
        //
        //----------------------------------------------------------------

        TmpFileEraserThunk tmpEraser(temporaryFilename, fileTools);

        {
            basic_ofstream<CharType> tmpStream(temporaryFilename.c_str(), ios_base::openmode(ios_base::out + ios_base::trunc));
            REQUIRE(!!tmpStream);

            String indent;

            if (tree.size() >= 1)
            {
                saveTree(indent, tree, tree[0], tmpStream);
            }

            REQUIRE(!!tmpStream);
        }

        //----------------------------------------------------------------
        //
        // Rename successfully updated file to the destination file.
        //
        //----------------------------------------------------------------

        REQUIRE(fileTools.renameFile(temporaryFilename.c_str(), filename));
        tmpEraser.cancel();

    }
    catch (const exception&) {REQUIRE(false);}

    returnTrue;
}

//================================================================
//
// FileEnvSTLImpl
//
//================================================================

class FileEnvSTLImpl : public FileEnv
{

public:

    stdbool loadFromFile(const CharType* filename, FileTools& fileTools, stdPars(ErrorLogKit))
        {return loadFile(filename, memory, fileTools, stdPassThru);}

    stdbool saveToFile(const CharType* filename, FileTools& fileTools, stdPars(ErrorLogKit)) const
        {return saveFile(memory, filename, fileTools, stdPassThru);}

    void eraseAll()
    {
        try {memory.clear();}
        catch (const exception&) {}
    }

    virtual bool get(const NameContainer& name, String& value, String& valueComment, String& blockComment) const
    {
        value.clear();
        valueComment.clear();
        blockComment.clear();

        try
        {
            Memory::const_iterator p = memory.find(name);

            ensure(p != memory.end());

            value = p->second.value;
            valueComment = p->second.valueComment;
            blockComment = p->second.blockComment;
        }
        catch (const exception&) {return false;}

        return true;
    }

    virtual bool set(const NameContainer& name, const String& value, const String& valueComment, const String& blockComment)
    {
        try
        {
            Memory::iterator p = memory.find(name);

            ////

            if (p != memory.end())
                memory.erase(p);

            Record rec;

            rec.value = value;
            rec.blockComment = blockComment;
            rec.valueComment = valueComment;

            memory.insert(pair<NameContainer, Record>(name, rec));
        }
        catch (const exception&) {return false;}

        return true;
    }

private:

    Memory memory;

};

//================================================================
//
// FileEnvSTL
//
//================================================================

FileEnvSTL::FileEnvSTL()
{
    FileEnvSTLImpl* p = 0;

    try {p = new (std::nothrow) FileEnvSTLImpl;}
    catch (const exception&) {}

    impl = p;
}

//----------------------------------------------------------------

FileEnvSTL::~FileEnvSTL()
{
    delete static_cast<FileEnvSTLImpl*>(impl);
}

//----------------------------------------------------------------

}
