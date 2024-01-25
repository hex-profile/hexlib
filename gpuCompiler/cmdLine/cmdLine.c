#include "cmdLine.h"

#include "parseTools/parseTools.h"

namespace cmdLine {

//================================================================
//
// readArgument
//
//================================================================

const CharType* readArgument(const CharType* ptr, const CharType* end, StlString& result)
{
    bool inString = false;

    for (; ;)
    {

        //
        // Normal segment, stops only on: special chars or interruptors (space, end)
        //

        const CharType* normBegin = ptr;

        if (inString)
        {
            while (ptr != end && *ptr != '\\' && *ptr != '"')
                ++ptr;
        }
        else
        {
            while (ptr != end && *ptr != '\\' && *ptr != '"' && !isSpaceTab(*ptr))
                ++ptr;
        }

        if (ptr != normBegin)
            result.append(normBegin, ptr);

        //
        // If interruptors, stop (inString cannot stop on space).
        //

        bool hasData = (ptr != end) && !isSpaceTab(*ptr);

        if_not (hasData)
            break;

        //
        // Quote or slash (advance guaranteed)
        //

        const CharType* slashBegin = ptr;

        while (ptr != end && *ptr == '\\')
            ++ptr;

        if (ptr != end && *ptr == '"')
        {
            size_t nslashes = ptr - slashBegin;
            result.append(slashBegin, nslashes >> 1);

            if (nslashes & 1)
                result.append("\"");
            else
                inString = !inString;

            ++ptr;
        }
        else
        {
            result.append(slashBegin, ptr); // pure slashes
        }
    }

    return ptr;
}

//================================================================
//
// parseCmdLine
//
// Parses according to MS rules, described in MSDN.
//
//================================================================

void parseCmdLine(const CharType* cmdLineBegin, const CharType* cmdLineEnd, vector<StlString>& appendList)
{

    const CharType* ptr = cmdLineBegin;
    const CharType* end = cmdLineEnd;

    //
    // Space/Argument interleave
    //

    for (; ;)
    {
        // Skip space (may not advance)
        skipSpaceTab(ptr, end);

        // Read argument (should advance)
        const CharType* argBegin = ptr;

        StlString s;
        ptr = readArgument(ptr, end, s);

        if (ptr == argBegin)
            break;

        appendList.push_back(s);
    }

}

//================================================================
//
// convertToArg
//
//================================================================

void convertToArg(const CharType* argBegin, const CharType* argEnd, StlString& result)
{
    const CharType* ptr = argBegin;
    const CharType* end = argEnd;

    result.append(CT("\""));

    ////

    for (; ;)
    {
        const CharType* normBegin = ptr;

        //
        // Normal block (may not move)
        //

        while (ptr != end && *ptr != '\\' && *ptr != '"')
            ++ptr;

        if (ptr != normBegin)
            result.append(normBegin, ptr);

        //
        // Slash or quote or end (move or exit guaranteed)
        //

        const CharType* slashBegin = ptr;

        while (ptr != end && *ptr == '\\')
            ++ptr;

        size_t nslashes = ptr - slashBegin;

        if (ptr == end)
        {
            result.append(2*nslashes, '\\');
            break;
        }
        else if (*ptr == '"') // ptr != end
        {
            result.append(2*nslashes, '\\');
            result.append(CT("\\\""));
            ++ptr;
        }
        else // some char
        {
            result.append(slashBegin, ptr);
        }
    }

    ////

    result.append(CT("\""));
}

//----------------------------------------------------------------

}
