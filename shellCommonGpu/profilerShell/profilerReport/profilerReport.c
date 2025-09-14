#include "profilerReport.h"

#include <algorithm>
#include <vector>
#include <map>
#include <time.h>

#include "cfgTools/numericVar.h"
#include "formattedOutput/sprintMsg.h"
#include "formattedOutput/textFiles.h"
#include "formatting/prettyNumber.h"
#include "interfaces/fileTools.h"
#include "numbers/float/floatType.h"
#include "numbers/mathIntrinsics.h"
#include "profilerShell/profiler/profilerTimer.h"
#include "simpleString/simpleString.h"
#include "storage/classThunks.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsgTrace.h"
#include "userOutput/paramMsg.h"
#include "formattedOutput/requireMsg.h"

namespace profilerReport {

using namespace std;

//================================================================
//
// SourceCache
//
//================================================================

using StringArray = vector<StlString>;

//----------------------------------------------------------------

struct SourceCache
{
    virtual void getFile(const StlString& filename, StringArray*& result, stdParsNull) =0;
};

//================================================================
//
// writeStylesheet
//
//================================================================

void writeStylesheet(const StlString& outputDirPrefix, stdPars(ReportKit))
{
    StlString thisFilename = outputDirPrefix + CT("profiler.css");

    OutputTextFile file;
    file.open(thisFilename.c_str(), stdPass);

    ////

    StlString cssText =
    # include "profiler.css"
    ;

    REQUIRE(cssText.size() >= 2);
    cssText = cssText.substr(1, cssText.size() - 2);

    file.write(cssText.data(), cssText.size());

    ////

    file.flushAndClose(stdPass);
}

//================================================================
//
// writeJavascript
//
//================================================================

void writeJavascript(OutputTextFile& file, stdPars(ReportKit))
{
    StlString jsText =
    # include "profiler.js"
    ;

    REQUIRE(jsText.size() >= 2);
    jsText = jsText.substr(1, jsText.size() - 2);

    file.write(jsText.data(), jsText.size());
}

//================================================================
//
// parseLocation
//
//================================================================

bool parseLocation(const StlString& location, StlString& filename, uint32& line, StlString& userMsg)
{
    const CharType* strBegin = location.c_str();
    const CharType* strEnd = strBegin + location.size();

    ////

    while (strEnd != strBegin && strEnd[-1] == ' ')
        --strEnd;

    while (strBegin != strEnd && *strBegin == ' ')
        ++strBegin;

    ////

    const CharType* msgBegin = strEnd;
    const CharType* msgEnd = strEnd;

    ////

    if (strEnd != strBegin && strEnd[-1] != ')')
    {
        const CharType* p = strEnd;

        while (p != strBegin && p[-1] != ':')
            --p;

        if (p != strBegin)
        {
            strEnd = p - 1;

            while (strEnd != strBegin && strEnd[-1] == ' ')
                --strEnd;

            msgBegin = p;

            while (msgBegin != msgEnd && *msgBegin == ' ')
                ++msgBegin;
        }
    }

    userMsg.assign(msgBegin, msgEnd);

    ////

    const CharType* p = strEnd;

    ensure(p != strBegin && p[-1] == ')');
    --p;

    ensure(p != strBegin && p[-1] >= '0' && p[-1] <= '9');
    uint32 number = p[-1] - '0';
    --p;

    uint32 multiplier = 10;

    while (p != strBegin && p[-1] >= '0' && p[-1] <= '9')
    {
        number += multiplier * (p[-1] - '0');
        --p; multiplier *= 10;
    }

    ////

    ensure(p != strBegin && p[-1] == '(');
    --p;

    while (p != strBegin && p[-1] == ' ')
        --p;

    const CharType* filenameEnd = p;

    ////

    const CharType* filenameBegin = strBegin;

    while (filenameEnd != filenameBegin && filenameEnd[-1] == ' ')
        --filenameEnd;

    while (filenameBegin != filenameEnd && *filenameBegin == ' ')
        ++filenameBegin;

    filename.assign(filenameBegin, filenameEnd);

    ////

    line = number;

    return true;
}

//================================================================
//
// STLSTRING_DECOMPOSE
//
//================================================================

#define STLSTRING_DECOMPOSE(str, prefix) \
    const CharType* prefix##Ptr = (str).c_str(); \
    const CharType* prefix##End = prefix##Ptr + (str).size();

//================================================================
//
// cutSimpleSurroundingBlock
//
//================================================================

void cutSimpleSurroundingBlock(const StringArray& text, Space rowIndex, Space linesUp, Space linesDown, StringArray& result)
{
    Space textSize = Space(text.size());

    Space indexOrg = clampRange(rowIndex - linesUp, 0, textSize);
    Space indexEnd = clampRange(rowIndex + 1 + linesDown, 0, textSize);

    result.clear();

    for (Space i = indexOrg; i < indexEnd; ++i)
        result.push_back(text[i]);
}

//================================================================
//
// getIndent
//
//================================================================

bool getIndent(const StlString& str, Space& result)
{
    STLSTRING_DECOMPOSE(str, str)
    const CharType* p = strPtr;

    while (p != strEnd && *p == ' ')
        ++p;

    result = Space(p - strPtr);
    return (p != strEnd);
}

//================================================================
//
// cutSmartBlock
//
//================================================================

void cutSmartBlock(const StringArray& text, Space rowIndex, StringArray& result, Space maxScanDistance, stdPars(ErrorLogKit))
{
    using namespace std;

    Space rowCount = Space(text.size());
    REQUIRE(rowIndex >= 0 && rowIndex < rowCount);

    ////

    Space scanMin = clampRange(rowIndex - maxScanDistance, 0, rowCount-1);
    Space scanMax = clampRange(rowIndex + maxScanDistance, 0, rowCount-1);

    ////

    Space lastRow = rowIndex;
    const CharType* finishPtr = 0;

    //----------------------------------------------------------------
    //
    // Forward-scan for the first ';'
    //
    //----------------------------------------------------------------

    //
    // Find the last ';' in the initial row
    //

    {
        STLSTRING_DECOMPOSE(text[rowIndex], str);

        const CharType* p = strEnd;

        while (p != strPtr && p[-1] != ';')
            --p;

        if (p != strPtr)
            finishPtr = p - 1;
    }

    //
    // If the start row is lone ')' bracket,
    // probably big macro
    //

    {
        STLSTRING_DECOMPOSE(text[rowIndex], str);

        const CharType* p = strEnd;

        while (p != strPtr && p[-1] == ' ')
            --p;

        if (p != strPtr && p[-1] == ')')
        {
            --p;
            const CharType* bracketPlace = p;

            while (p != strPtr && p[-1] == ' ')
                --p;

            if (p == strPtr)
                finishPtr = bracketPlace;
        }
    }

    //----------------------------------------------------------------
    //
    // If not found in the first row, scan FORWARD for an enclosing ';'
    //
    //----------------------------------------------------------------

    if_not (finishPtr)
    {
        for (lastRow = rowIndex + 1; lastRow <= scanMax; ++lastRow)
        {
            STLSTRING_DECOMPOSE(text[lastRow], str);

            const CharType* p = strPtr;
            while (p != strEnd && *p != ';')
                ++p;

            if (p != strEnd)
                {finishPtr = p; break;}

            ////

            p = strEnd;

            while (p != strPtr && p[-1] != ')')
                --p;

            if (p != strPtr)
            {
                --p;

                while (p != strPtr && p[-1] == ' ')
                    --p;

                if (p == strPtr) // lone ')'
                    {finishPtr = p-1; break;}
            }
        }
    }

    ////

    require(finishPtr != 0); // found?

    //----------------------------------------------------------------
    //
    // Scan BACKWARD to match the number of '(' and ')'
    //
    //----------------------------------------------------------------

    bool openParenthesisIsFirstInRow = false;
    Space parenthesisLevel = 0;

    ////

    Space firstRow = lastRow;

    for (; firstRow >= scanMin; --firstRow)
    {
        STLSTRING_DECOMPOSE(text[firstRow], str);
        const CharType* p = strEnd;
        if (firstRow == lastRow) p = finishPtr+1;

        for (; ;)
        {
            while (p != strPtr && p[-1] != '(' && p[-1] != ')')
                --p;

            if (p == strPtr)
                break;

            CharType c = *--p;
            if (c == ')') ++parenthesisLevel;
            if (c == '(') --parenthesisLevel;

            if (parenthesisLevel == 0)
                break;
        }

        ////

        if (parenthesisLevel <= 0)
        {
            while (p != strPtr && p[-1] == ' ')
                --p;

            openParenthesisIsFirstInRow = (p == strPtr);

            break;
        }
    }

    ////

    require(firstRow >= scanMin); // found?

    //
    // Slipped: detection failure
    //

    if_not (rowIndex >= firstRow && rowIndex <= lastRow)
        returnFalse;

    //----------------------------------------------------------------
    //
    // Cut the text block
    //
    //----------------------------------------------------------------

    if (openParenthesisIsFirstInRow)
        firstRow = clampRange(firstRow - 1, 0, rowCount-1); // one row up

    result.clear();

    for (Space i = firstRow; i <= lastRow; ++i)
        result.push_back(text[i]);
}

//================================================================
//
// removeIndent
//
//================================================================

void removeIndent(StringArray& result)
{
    size_t minIndent = spaceMax;

    ////

    for (StringArray::const_iterator q = result.begin(); q != result.end(); ++q)
    {
        STLSTRING_DECOMPOSE(*q, str);

        const CharType* p = strPtr;
        while (p != strEnd && *p == ' ') ++p;

        if (p != strEnd)
        {
            size_t indent = p - strPtr;
            minIndent = minv(minIndent, indent);
        }
    }

    ////

    if (minIndent != spaceMax)
    {
        for (StringArray::iterator q = result.begin(); q != result.end(); ++q)
        {
            size_t n = clampMax(minIndent, q->size());
            *q = q->substr(n);
        }
    }
}

//================================================================
//
// trimEmptyLines
//
//================================================================

void trimEmptyLines(StringArray& result)
{
    StringArray::iterator i = result.begin();

    while (i != result.end() && i->size() == 0)
        ++i;

    if (i != result.begin())
        result.erase(result.begin(), i);

    ////

    i = result.end();

    while (i != result.begin() && i[-1].size() == 0)
        --i;

    if (i != result.end())
        result.erase(i, result.end());
}

//================================================================
//
// stripFilePath
//
//================================================================

StlString stripFilePath(const StlString& str)
{
    STLSTRING_DECOMPOSE(str, str);

    const CharType* p = strEnd;

    while (p != strPtr && p[-1] != '\\' && p[-1] != '/')
        --p;

    return StlString(p, strEnd);
}

//================================================================
//
// CodeLocation
//
//================================================================

struct CodeLocation
{

public:

    StlString filename;
    uint32 lineNumber = 0;
    StlString userMsg;

    StringArray code;

public:

    void clear()
    {
        filename.clear();
        lineNumber = 0;
        userMsg.clear();
        code.clear();
    }
};

//================================================================
//
// locationMsg
//
//================================================================

template <typename Kit>
StlString locationMsg(const Kit& kit, const CodeLocation& l)
{
    StlString result;

    if (l.filename.size())
        result = sprintMsg(kit, STR("%0(%1)"), l.filename, l.lineNumber);

    if (l.userMsg.size())
    {
        if (result.size()) result += CT(": ");
        result += l.userMsg;
    }

    return result;
}

//================================================================
//
// trimRightSpaces
//
//================================================================

StlString trimRightSpaces(const StlString& str)
{
    STLSTRING_DECOMPOSE(str, str);

    const CharType* p = strEnd;

    while (p != strPtr && p[-1] == ' ')
        --p;

    return StlString(strPtr, p);
}

//================================================================
//
// CodeBlockParams
//
//================================================================

struct CodeBlockParams
{
    Space simpleCutRadius;
    Space smartMaxScanRows;
    Space smartMaxRows;
};

//================================================================
//
// getCodeBlockCore
//
//================================================================

void getCodeBlockCore(const StlString& location, const CodeBlockParams& o, SourceCache& sourceCache, CodeLocation& result, stdPars(ReportKit))
{
    result.clear();

    ////

    StlString filename;
    uint32 lineNumber;
    StlString userMsg;

    ////

    if_not (parseLocation(location, filename, lineNumber, userMsg))
    {
        printMsg(kit.msgLog, STR("Failed to parse location %0"), location, msgErr);
        returnFalse;
    }

    //
    // Load code block
    //

    StringArray* arr = 0;
    sourceCache.getFile(filename, arr, stdPass);
    REQUIRE(arr != 0);
    StringArray& text = *arr;

    ////

    REQUIRE(lineNumber < spaceMax);
    Space rowIndex = clampRange(Space(lineNumber) - 1, 0, Space(arr->size()));

    ////

    bool smartSuccess = errorBlock(cutSmartBlock(text, rowIndex, result.code, o.smartMaxScanRows, stdPassNc));

    if_not (smartSuccess)
        cutSimpleSurroundingBlock(text, rowIndex, o.simpleCutRadius, o.simpleCutRadius, result.code);

    ////

    removeIndent(result.code);
    trimEmptyLines(result.code);

    ////

    if (smartSuccess && o.smartMaxRows >= 1 && result.code.size() > size_t(o.smartMaxRows))
    {
        result.code.resize(o.smartMaxRows - 1);
        trimEmptyLines(result.code);

        Space indent = 0;

        if (result.code.size())
        {
            if_not (getIndent(result.code.back(), indent))
                indent = 0;

            // result.code.push_back(StlString(indent, ' ') + CT(".."));
        }
    }

    ////

    result.filename = stripFilePath(filename);
    result.lineNumber = lineNumber;
    result.userMsg = userMsg;
}

//================================================================
//
// getCodeBlock
//
//================================================================

void getCodeBlock(const StlString& location, const CodeBlockParams& o, SourceCache& sourceCache, CodeLocation& result, stdPars(ReportKit))
{
    if_not (errorBlock(getCodeBlockCore(location, o, sourceCache, result, stdPassThruNc)))
    {
        result.clear();
        result.userMsg = location;
        result.code.push_back(CT("(failed to get file contents)"));
    }
}

//================================================================
//
// translateStringToHtml
//
//================================================================

StlString translateStringToHtml(const StlString& str, size_t maxLength)
{
    StlString result;

    ////

    StlString s = str;
    bool truncated = false;

    size_t truncSymbolSize = 2;

    ////

    if (maxLength >= truncSymbolSize && s.size() > maxLength)
    {
        s = s.substr(0, maxLength - truncSymbolSize); // reserve 2 chars for ellipsis
        truncated = true;
    }

    ////

    const CharType* strBegin = s.c_str();
    const CharType* strEnd = strBegin + s.size();

    ////

    for (const CharType* p = strBegin; ;)
    {
        const CharType* start = p;

        ////

        while (p != strEnd && *p != '&' && *p != '"'&& *p != '\'' && *p != '<' && *p != '>')
            ++p;

        if (p != start)
            result.append(start, p);

        ////

        if (p == strEnd)
            break;

        if (*p == '&') result.append(CT("&amp;"));
        if (*p == '"') result.append(CT("&quot;"));
        if (*p == '\'') result.append(CT("&apos;"));
        if (*p == '<') result.append(CT("&lt;"));
        if (*p == '>') result.append(CT("&gt;"));

        ++p;
    }

    ////

    if (truncated)
        result += CT("..");

    ////

    return result;
}

//================================================================
//
// Timing
//
//================================================================

struct Timing
{
    float32 totalTime; // per cycle
    float32 repetitionFactor; // avg number of calls per cycle

    float32 avgElemCount;
    float32 avgElemClocks;

    float32 avgPredictedOverhead; // per cycle
};

//================================================================
//
// TimingParams
//
//================================================================

struct TimingParams
{
    bool deviceTimingMode;
    float32 tickFactor;
    float32 divCycleCount;
    float32 processingThroughput;
};

//================================================================
//
// computeNodeTiming
//
//================================================================

void computeNodeTiming(const ProfilerNode& node, const TimingParams& o, Timing& info)
{
    float32 avgRunFactor = float32(node.counter) * o.divCycleCount;

    ////

    float32 nodeTotalTime = o.deviceTimingMode ?
        float32(node.deviceTotalTime) :
        float32(node.totalTimeSum) * o.tickFactor;

    info.totalTime = nodeTotalTime * o.divCycleCount;

    ////

    info.repetitionFactor = float32(node.counter) * o.divCycleCount;

    ////

    float32 avgElemClocks = 0;
    float32 avgElemCount = 0;

    if (node.totalElemCount)
    {
        avgElemCount = float32(node.totalElemCount) * fastRecipZero(float32(node.counter));

        float32 divNodeTotalElemCount = fastRecipZero(float32(node.totalElemCount));
        float32 avgElemTime = nodeTotalTime * divNodeTotalElemCount;

        avgElemClocks = avgElemTime * o.processingThroughput;
        if_not (def(avgElemClocks)) avgElemClocks = 0;
    }

    info.avgElemClocks = avgElemClocks;
    info.avgElemCount = avgElemCount;

    ////

    float32 totalOverheadTime = o.deviceTimingMode ?
        float32(node.deviceTotalOverheadTime) : 0;

    info.avgPredictedOverhead = totalOverheadTime * o.divCycleCount;
}

//================================================================
//
// NodeHash
//
//================================================================

struct NodeHash
{
    uint32 A;
    uint32 B;

    sysinline NodeHash() {A = B = 0;}
};

//================================================================
//
// updateHash
//
//================================================================

void updateHash(NodeHash& hash, const CharType* ptr)
{
    uint32 A = hash.A;
    uint32 B = hash.B;

    ////

    for (CharType c = *ptr; c != 0; c = *++ptr)
    {
        uint32 v = uint32(c);

        A = A * 101 + v;
        B = B * 107 + v;
    }

    hash.A = A;
    hash.B = B;
}

//================================================================
//
// updateHash
//
//================================================================

void updateHash(NodeHash& hash, const CodeLocation& location)
{
    if (location.userMsg.size())
        updateHash(hash, location.userMsg.c_str());

    for (StringArray::const_iterator i = location.code.begin(); i != location.code.end(); ++i)
        updateHash(hash, i->c_str());
}

//================================================================
//
// DEBUG_CALLSTACK
//
//================================================================

#define DEBUG_CALLSTACK 0

////

#if DEBUG_CALLSTACK
    #define DEBUG_CALLSTACK_ONLY(x) x
#else
    #define DEBUG_CALLSTACK_ONLY(x)
#endif

//================================================================
//
// NodeInfo
//
//================================================================

struct NodeInfo
{
    Timing timing;
    NodeHash hash;
    DEBUG_CALLSTACK_ONLY(vector<StlString> stack;)
    StlString filename;
    vector<CodeLocation> locations;
    bool expandFlag = false;
};

//================================================================
//
// SortComparator
//
//================================================================

class SortComparator
{

public:

    SortComparator(const NodeInfo* nodeArray)
        : nodeArray(nodeArray) {}

    inline bool operator () (Space a, Space b)
    {
        return nodeArray[a].timing.totalTime > nodeArray[b].timing.totalTime;
    }

private:

    const NodeInfo* nodeArray;

};

//================================================================
//
// factorIsEqual
//
//================================================================

inline bool factorIsEqual(float32 a, float32 b, float32 tolerance)
{
    return
        b >= (1 - tolerance) * a &&
        b <= (1 + tolerance) * a;
}

//================================================================
//
// DisplayParams
//
//================================================================

struct DisplayParams
{
    float32 timeThresholdFraction;
    float32 timeThresholdMin;
    float32 timeThresholdMax;
    int32 timeMsDigits;
    float32 factorTolerance;
    int32 factorDigits;
    int32 maxSourceLineLength;
    uint32 maxCallstackFullyExpanded;
    uint32 minCallstackExpanded;
};

//================================================================
//
// HtmlReportParams
//
//================================================================

struct HtmlReportParams
{
    const TimingParams& timingParams;
    const CodeBlockParams& codeBlockParams;
    const DisplayParams& displayParams;
    const StlString& outputDirPrefix;
    const StlString& reportCreationTime;
    SourceCache& sourceCache;
};

//================================================================
//
// getTimeDigits
//
//================================================================

int32 getTimeDigits(float32 value, int32 minDigits, int32 accuracyDigits)
{
    float32 absVal = absv(value);

    return
        absVal == 0 ? minDigits :
        clampMin(accuracyDigits + convertUp<int32>(-log10f(value)) - 1, minDigits);
}

//----------------------------------------------------------------

inline FormatNumber<float32> formatTime(const float32& value, const DisplayParams& params)
    {return formatNumber(value * 1000.f, FormatNumberOptions().fformF().precision(params.timeMsDigits));}

//================================================================
//
// generateHtmlForTree
//
//================================================================

void generateHtmlForTree(const ProfilerNode& thisNode, const NodeInfo& thisInfo, const StlString& parentHtml, const HtmlReportParams& o, stdPars(ReportKit))
{
    using namespace std;

    float32 timeThresholdFraction = o.displayParams.timeThresholdFraction;
    float32 timeThresholdMin = o.displayParams.timeThresholdMin;
    float32 timeThresholdMax = o.displayParams.timeThresholdMax;

    float32 factorTolerance = o.displayParams.factorTolerance;
    int32 factorDigits = o.displayParams.factorDigits;

    ////

    float32 thisNodeTimeThreshold = clampRange(timeThresholdFraction * thisInfo.timing.totalTime, timeThresholdMin, timeThresholdMax);

    //----------------------------------------------------------------
    //
    // Childs pointer array (in execution order)
    //
    //----------------------------------------------------------------

    Space childrenCount = 0;

    for (ProfilerNode* p = thisNode.lastChild; p != 0; p = p->prevBrother)
        ++childrenCount;

    ////

    vector<ProfilerNode*> children(childrenCount);

    ////

    Space childIdx = childrenCount;

    for (ProfilerNode* p = thisNode.lastChild; p != 0; p = p->prevBrother)
        children[--childIdx] = p;

    REQUIRE(childIdx == 0);

    //----------------------------------------------------------------
    //
    // * Detect and unfold "thunk" nodes on children.
    // * Compute info for all children.
    //
    //----------------------------------------------------------------

    vector<NodeInfo> nodeInfo(childrenCount);

    bool makeThunkUnfolding = true;

    ////

    for_count (i, childrenCount)
    {

        ProfilerNode* node = children[i];
        NodeInfo& info = nodeInfo[i];

        computeNodeTiming(*node, o.timingParams, info.timing);

        float32 originalTotalTime = info.timing.totalTime;
        float32 maxLostTime = clampRange(originalTotalTime * timeThresholdFraction, timeThresholdMin, timeThresholdMax);

        ////

        {
            CodeLocation code;
            errorBlock(getCodeBlock(node->location, o.codeBlockParams, o.sourceCache, code, stdPassNc));
            info.locations.push_back(code);

            ////

            info.hash = thisInfo.hash;
            updateHash(info.hash, code); // cumulative hash
        }

        ////

        DEBUG_CALLSTACK_ONLY(info.stack = thisInfo.stack);
        DEBUG_CALLSTACK_ONLY(info.stack.push_back(node->location));

        ////

        if (makeThunkUnfolding)
        {
            for (; ;)
            {
                float32 maxChildTime = 0;
                ProfilerNode* maxChild = 0;

                for (ProfilerNode* p = node->lastChild; p != 0; p = p->prevBrother)
                {
                    Timing timing;
                    computeNodeTiming(*p, o.timingParams, timing);

                    if (timing.totalTime >= maxChildTime)
                        {maxChildTime = timing.totalTime; maxChild = p;}
                }

                ////

                float32 lostTime = clampMin(originalTotalTime - maxChildTime, 0.f);

                if_not (maxChild && lostTime <= maxLostTime)
                    break;

                ////

                node = maxChild;

                ////

                DEBUG_CALLSTACK_ONLY(info.stack.push_back(maxChild->location));

                ////

                {
                    CodeLocation code;
                    errorBlock(getCodeBlock(maxChild->location, o.codeBlockParams, o.sourceCache, code, stdPassNc));
                    info.locations.push_back(code);

                    updateHash(info.hash, code); // cumulative hash
                }

            }
        }

        ////

        if (node != children[i])
        {
            computeNodeTiming(*node, o.timingParams, info.timing);
            children[i] = node;
        }

        ////

        float32 totalChildrenTime = 0;

        for (ProfilerNode* p = node->lastChild; p != 0; p = p->prevBrother)
        {
            Timing timing;
            computeNodeTiming(*p, o.timingParams, timing);

            totalChildrenTime += timing.totalTime;
        }

        ////

        info.filename = sprintMsg(kit, STR("%0%1.html"), hex(info.hash.A), hex(info.hash.B));

        ////

        info.expandFlag =
            (node->lastChild != 0) &&
            (info.timing.totalTime > thisNodeTimeThreshold) &&
            (totalChildrenTime > maxLostTime);

    }

    //----------------------------------------------------------------
    //
    // HTML header
    //
    //----------------------------------------------------------------

    StlString thisFilename = o.outputDirPrefix + thisInfo.filename;

    OutputTextFile file;
    file.open(thisFilename.c_str(), stdPass);

    auto log = getLog(file, kit);

    {

        printMsg(log, STR("<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">"));

        printMsg(log, STR("<html>"));
        REMEMBER_CLEANUP(printMsg(log, STR("</html>")));

        ////

        {
            printMsg(log, STR("<head>"));
            REMEMBER_CLEANUP(printMsg(log, STR("</head>")));

            printMsg(log, STR("<title>Profiler Report %0</title>"), o.reportCreationTime);

            printMsg(log, STR("<link href=\"profiler.css\" rel=\"stylesheet\" type=\"text/css\"/>"));
        }

        printMsg(log, STR("<body>"));
        REMEMBER_CLEANUP(printMsg(log, STR("</body>")));

        //----------------------------------------------------------------
        //
        // Javascript
        //
        //----------------------------------------------------------------

        writeJavascript(file, stdPass);

        //----------------------------------------------------------------
        //
        // Table header
        //
        //----------------------------------------------------------------

        printMsg(log, '$', STR("<pre class=\"headerBlock\"><a href=\"index.html\">Profiler report created at $, $, Minimal displayed fraction $%</a></pre>"),
            o.reportCreationTime, o.timingParams.deviceTimingMode ? STR("Device mode") : STR("Host mode"),
            fltf(o.displayParams.timeThresholdFraction * 100, 1));

        //----------------------------------------------------------------
        //
        // Debug call stack
        //
        //----------------------------------------------------------------

        uint32 clickId = 0;

        ////

        #if DEBUG_CALLSTACK

            {
                printMsg(log, STR("<a id=\"I%0\" title=\"%1\" onclick=\"toggleBlock(this);\" href=\"#_I%0\"> <pre class=\"linkBlock\">"),
                    hex(clickId), STR("Click to expand the block (Javascript must be enabled)"));

                printMsg(log, STR("(show full callstack)"));

                printMsg(log, STR("</pre></a>"));

                printMsg(log, STR("<pre id=\"I%0_\" style=\"display:none\">"), hex(clickId));

                ////

                for_count (i, thisInfo.stack.size())
                    printMsg(log, STR("%0"), thisInfo.stack[i]);

                printMsg(log, STR(""));

                ////

                printMsg(log, STR("</pre>"));

                ++clickId;
            }

        #endif

        ////

        printMsg(log, STR("<pre>"));
        printMsg(log, STR(""));
        printMsg(log, STR("</pre>"));

        ////

        printMsg(log, STR("<table>"));
        REMEMBER_CLEANUP(printMsg(log, STR("</table>")));


        {
            printMsg(log, STR("<tr>"));
            REMEMBER_CLEANUP(printMsg(log, STR("</tr>")));

            printMsg(log, STR("<th>Name</th>"));
            printMsg(log, STR("<th>Time (Per Frame)</th>"));
        }

        //----------------------------------------------------------------
        //
        // UP and total time
        //
        //----------------------------------------------------------------

        float32 totalChildrenTime = 0;

        for_count (i, childrenCount)
            totalChildrenTime += nodeInfo[i].timing.totalTime;

        float32 thisNodeBodyTime = clampMin(thisInfo.timing.totalTime - totalChildrenTime, 0.f);

        ////

        {
            printMsg(log, STR("<tr class=\"nodeTotal\">"));
            REMEMBER_CLEANUP(printMsg(log, STR("</tr>")));

            ////

            if (parentHtml.size())
                printMsg(log, STR("<td class = \"uplinkTd\"><a href=\"%0\"><pre class=\"linkBlock\">"), parentHtml);
            else
                printMsg(log, STR("<td> <pre>"));

            printMsg(log, STR(""));
            printMsg(log, parentHtml.size() ? STR("&uarr; Up &uarr;") : STR(""));
            printMsg(log, STR(""));
            printMsg(log, STR("</pre>%0</td>"), parentHtml.size() ? STR("</a>") : STR(""));

            ////

            printMsg
            (
                log,

                !(thisNodeBodyTime > thisNodeTimeThreshold) ?
                    STR("<td>%0 ms</td>") :
                    STR("<td>%0 ms, with %1 ms body</td>"),

                formatTime(thisInfo.timing.totalTime, o.displayParams),
                formatTime(thisNodeBodyTime, o.displayParams)
            );
        }

        //----------------------------------------------------------------
        //
        //
        //
        //----------------------------------------------------------------

        vector<Space> displayOrder(childrenCount);

        for_count (i, childrenCount)
            displayOrder[i] = i;

        if (childrenCount)
            sort(displayOrder.begin(), displayOrder.end(), SortComparator(&nodeInfo[0]));

        //----------------------------------------------------------------
        //
        // Report children
        //
        //----------------------------------------------------------------

        for_count (j, childrenCount)
        {
            Space i = displayOrder[j];

            ////

            const NodeInfo& info = nodeInfo[i];

            if_not (info.timing.totalTime > thisNodeTimeThreshold)
                continue; // Skip the child

            ////

            printMsg(log, STR("<tr>"));
            REMEMBER_CLEANUP(printMsg(log, STR("</tr>")));

            const ProfilerNode& node = *children[i];

            bool makeLink = info.expandFlag;

            printMsg(log, STR("<td class=\"nameCell\">"));

            ////

            size_t unfoldCounter = info.locations.size();
            vector<CodeLocation>::const_reverse_iterator locationsBegin = info.locations.rbegin();
            vector<CodeLocation>::const_reverse_iterator locationsEnd = info.locations.rend();
            vector<CodeLocation>::const_reverse_iterator k = locationsBegin;

            if (info.locations.size() > o.displayParams.maxCallstackFullyExpanded)
                unfoldCounter = o.displayParams.minCallstackExpanded;

            ////

            printMsg(log, !makeLink ? STR("<pre>") : STR("<a href=\"%0\"> <pre class=\"linkBlock\">"), info.filename);
            printMsg(log, STR(""));

            for (; k != locationsEnd && unfoldCounter != 0; ++k, --unfoldCounter)
            {
                if (k != locationsBegin)
                {
                    printMsg(log, STR(""));
                    printMsg(log, STR("<hr>"));
                }

                for (StringArray::const_iterator i = k->code.begin(); i != k->code.end(); ++i)
                    printMsg(log, STR("%0"), translateStringToHtml(*i, o.displayParams.maxSourceLineLength));

                printMsg(log, STR(""));
                printMsg(log, STR("%0%1"), locationMsg(kit, *k), k+1 == locationsEnd ? STR("") : STR(", called from:"));
            }

            ////

            if (k == locationsEnd)
                printMsg(log, STR(""));

            printMsg(log, !makeLink ? STR("</pre>") : STR("</pre> </a>"));

            ////

            if (k != locationsEnd)
            {
                printMsg(log, STR("<a id=\"I%0\" title=\"%1\" onclick=\"toggleBlock(this);\" href=\"#_I%0\"> <pre class=\"linkBlock\">"),
                    hex(clickId), STR("Click to expand the block (Javascript must be enabled)"));

                printMsg(log, STR(""));
                printMsg(log, STR("(more...)"));

                printMsg(log, STR("</pre></a>"));

                printMsg(log, STR("<pre id=\"I%0_\" style=\"display:none\">"), hex(clickId));

                vector<CodeLocation>::const_reverse_iterator startIter = k;

                for (; k != locationsEnd; ++k)
                {
                    {
                        if (k != startIter) printMsg(log, STR(""));
                        printMsg(log, STR("<hr>"));
                    }

                    for (StringArray::const_iterator i = k->code.begin(); i != k->code.end(); ++i)
                        printMsg(log, STR("%0"), translateStringToHtml(*i, o.displayParams.maxSourceLineLength));

                    printMsg(log, STR(""));
                    printMsg(log, STR("%0%1"), locationMsg(kit, *k), k+1 == locationsEnd ? STR("") : STR(", called from:"));
                }

                ////

                printMsg(log, STR(""));

                ////

                printMsg(log, STR("</pre>"));

                ++clickId;
            }

            ////

            printMsg(log, STR("</td>"));

            ////

            const Timing& timing = info.timing;

            float32 repetitionFactor = timing.repetitionFactor;
            float32 parentFactor = thisInfo.timing.repetitionFactor;

            printMsg(log, STR("<td>"));

            ////

            StlString repetitionMsg;

            if_not (factorIsEqual(repetitionFactor, 1, factorTolerance))
            {
                float32 nestingFactor = timing.repetitionFactor / parentFactor;
                int32 nestingFactorDigits = factorIsEqual(floorv(nestingFactor + 0.5f), nestingFactor, factorTolerance) ? 0 : factorDigits;

                repetitionMsg = sprintMsg
                (
                    kit,
                    factorIsEqual(parentFactor, 1, factorTolerance) ?
                        STR("%1 &times;") :

                    factorIsEqual(nestingFactor, 1, factorTolerance) ?
                        STR("%0 &times;") :

                    STR("%0 &times; %1 &times;"),

                    fltf(parentFactor, factorDigits),
                    fltf(nestingFactor, nestingFactorDigits)
                );
            }

            ////

            printMsg
            (
                log,
                !repetitionMsg.size() ? STR("<p>%0 ms</p>") : STR("<p>%1 %2 ms = %0 ms</p>"),
                formatTime(timing.totalTime, o.displayParams),
                repetitionMsg,
                formatTime(timing.totalTime / repetitionFactor, o.displayParams)
            );

            ////

            if (timing.avgElemClocks > 0)
                printMsg(log, STR("<p>%0 clocks &times; %1 elements</p>"),
                    prettyNumber(timing.avgElemClocks), prettyNumber(timing.avgElemCount));

            ////

            if (timing.avgPredictedOverhead > thisNodeTimeThreshold)
            {
                StlString timeMsg = sprintMsg
                (
                    kit,
                    !repetitionMsg.size() ? STR("~%0 ms") : STR("%1 %2 ms = ~%0 ms"),
                    formatTime(timing.avgPredictedOverhead, o.displayParams),
                    repetitionMsg,
                    formatTime(timing.avgPredictedOverhead / repetitionFactor, o.displayParams)
                );

                printMsg(log, STR("<p>Potential overhead up to %0 </p>"), timeMsg);
            }

            printMsg(log, STR("</td>"));
        }

    }

    ////

    file.flushAndClose(stdPass);

    //----------------------------------------------------------------
    //
    // Recursive call
    //
    //----------------------------------------------------------------

    for_count (i, childrenCount)
    {
        const ProfilerNode& node = *children[i];
        const NodeInfo& info = nodeInfo[i];

        if (info.expandFlag)
            generateHtmlForTree(node, info, thisInfo.filename, o, stdPass);
    }
}

//================================================================
//
// SourceCacheImpl
//
//================================================================

class SourceCacheImpl
{

public:

    using Kit = DiagnosticKit;

public:

    void getFile(const StringArray& searchPath, const StlString& filename, StringArray*& result, stdPars(Kit));

private:

    struct FileCache
    {
        bool ok = false;
        StringArray content;
    };

    using Memory = map<StlString, FileCache>;
    Memory memory;

};

//================================================================
//
// SourceCacheImpl::getFile
//
//================================================================

void SourceCacheImpl::getFile(const StringArray& searchPath, const StlString& filename, StringArray*& result, stdPars(Kit))
{
    using namespace std;

    //
    // Try to find, if not, make empty file;
    //
    // If a file fails to open, the next calls with its name return empty list without file operations.
    //

    pair<Memory::iterator, bool> seachResult = memory.insert(make_pair(filename, FileCache{}));
    auto& searchValue = seachResult.first->second;

    result = &searchValue.content;

    if (!seachResult.second)
    {
        require(searchValue.ok);
        return;
    }

    //
    // Try to read file
    //

    basic_ifstream<CharType> stream;

    for (auto i : searchPath)
    {
        auto fullName = i + filename;

        stream.clear();
        stream.open(fullName.c_str());

        if (stream)
            break;
    }

    ////

    REQUIRE_MSG1(!!stream, STR("Cannot find file %0"), filename);

    ////

    for (; ;)
    {
        StlString s;
        getline(stream, s);

        if (!stream)
            break;

        searchValue.content.push_back(s);
    }

    ////

    REQUIRE_TRACE1(stream.eof(), STR("Cannot read file %0"), filename);

    ////

    searchValue.ok = true;
}

//================================================================
//
// SourceCacheThunk
//
//================================================================

class SourceCacheThunk : public SourceCache
{

public:

    void getFile(const StlString& filename, StringArray*& result, stdParsNull)
        {impl.getFile(searchPath, filename, result, stdPassThru);}

public:

    using Kit = DiagnosticKit;

    SourceCacheThunk(SourceCacheImpl& impl, const StringArray& searchPath, const Kit& kit)
        : impl(impl), searchPath(searchPath), kit(kit) {}

private:

    SourceCacheImpl& impl;
    const StringArray& searchPath;
    Kit kit;

};

//================================================================
//
// HtmlReportImpl
//
//================================================================

class HtmlReportImpl
{

public:

    HtmlReportImpl();
    void serialize(const CfgSerializeKit& kit);
    void makeReport(const MakeReportParams& o, stdPars(ReportFileKit));

private:

    SourceCacheImpl sourceCache;

    NumericVarStatic<int32, 0, 256, 7> simpleMaxRows;
    int32 smartMaxScanRows = 64;
    NumericVarStatic<int32, 1, 256, 6> smartMaxRows;

    NumericVarStaticEx<float32, int32, 0, 100, 0> timeThresholdParentFractionInPercents;
    NumericVarStaticEx<float32, int32, 0, 0xFFFF, 0> timeThresholdSignificantTimeInMs;
    NumericVarStatic<int32, 0, 8, 3> timeMsDigits;

    NumericVarStatic<int32, 0, 8, 2> factorDigits;

    NumericVarStatic<int32, 16, 65536, 96> maxSourceLineLength;
    NumericVarStatic<uint32, 1, 32, 3> maxCallstackFullyExpanded;
    NumericVarStatic<uint32, 1, 32, 2> minCallstackExpanded;

};

//----------------------------------------------------------------

CLASSTHUNK_CONSTRUCT_DESTRUCT(HtmlReport)
CLASSTHUNK_VOID1(HtmlReport, serialize, const CfgSerializeKit&)
CLASSTHUNK_BOOL_STD1(HtmlReport, makeReport, const MakeReportParams&, ReportFileKit)

//================================================================
//
// HtmlReportImpl::HtmlReportImpl
//
//================================================================

HtmlReportImpl::HtmlReportImpl()
{
    timeThresholdParentFractionInPercents = 0.5f;
    timeThresholdSignificantTimeInMs = 0.1f;
}

//================================================================
//
// HtmlReportImpl::serialize
//
//================================================================

void HtmlReportImpl::serialize(const CfgSerializeKit& kit)
{
    simpleMaxRows.serialize(kit, STR("Simple Cut: Max Rows"));
    smartMaxRows.serialize(kit, STR("Smart Cut: Max Rows"));

    {
        CFG_NAMESPACE("Displayed Time Thresholds");

        timeThresholdParentFractionInPercents.serialize(kit, STR("Fraction Of Parent To Be Hidden (In Percents)"));
        timeThresholdSignificantTimeInMs.serialize(kit, STR("Max Hidden Time In Milliseconds"), STR(""));
        timeMsDigits.serialize(kit, STR("Fractional Digits Of Millisecond"));
    }

    factorDigits.serialize(kit, STR("Factor Output Digits"));
    maxSourceLineLength.serialize(kit, STR("Max Source Line Length"));
    maxCallstackFullyExpanded.serialize(kit, STR("Max Callstack Fully Expanded"));
    minCallstackExpanded.serialize(kit, STR("Min Callstack Expanded"));
}

//================================================================
//
// HtmlReportImpl::makeReport
//
//================================================================

void HtmlReportImpl::makeReport(const MakeReportParams& o, stdPars(ReportFileKit))
{
    REQUIRE(o.cycleCount >= 0);
    require(o.cycleCount >= 1);

    ////

    fileTools::makeDirectory(o.outputDir);

    ////

    try
    {

        //
        // Accumulate root node.
        //

        ProfilerNode* rootNode = o.rootNode;
        REQUIRE(rootNode);

        rootNode->totalTimeSum = 0;

        for (ProfilerNode* p = rootNode->lastChild; p != 0; p = p->prevBrother)
        {
            rootNode->totalTimeSum += p->totalTimeSum;
        }

        rootNode->counter = o.cycleCount;

        //
        // Accumulate subtree device time
        //

        profilerUpdateDeviceTreeTime(*rootNode);
        bool deviceTimingMode = rootNode->deviceTotalTime > 0;

        //
        // Print tree
        //

        REQUIRE(o.divTicksPerSec > 0);
        float32 tickFactor = o.divTicksPerSec;

        ////

        time_t rawTime;
        REQUIRE(time(&rawTime) != -1);
        tm* locTime = localtime(&rawTime);
        REQUIRE(locTime != 0);

        StlString timeStr = sprintMsg(kit, STR("%0:%1:%2"), dec(locTime->tm_hour, 2), dec(locTime->tm_min, 2), dec(locTime->tm_sec, 2));

        ////

        StringArray searchPath;
        searchPath.push_back("");

        SourceCacheThunk sourceCacheThunk(sourceCache, searchPath, kit);

        ////

        TimingParams timingParams{deviceTimingMode, tickFactor, 1.f/o.cycleCount, o.processingThroughput};

        NodeInfo rootInfo;
        computeNodeTiming(*rootNode, timingParams, rootInfo.timing);
        rootInfo.hash = NodeHash();
        rootInfo.filename = CT("index.html");

        CodeLocation rootLocation;
        rootLocation.userMsg = CT("<Profiler Root>");
        rootInfo.locations.push_back(rootLocation);

        ////

        StlString outputDirPrefix = StlString(o.outputDir) + CT("/");
        writeStylesheet(outputDirPrefix, stdPass);

        CodeBlockParams codeBlockParams{(simpleMaxRows | 1) / 2, smartMaxScanRows, smartMaxRows};

        float32 timeThresholdFraction = timeThresholdParentFractionInPercents * 0.01f;
        float32 timeThresholdMin = powf(10.f, -float32(timeMsDigits + 3)) / 2; // Minimal visible time with given precision.
        float32 timeThresholdMax = clampMin(timeThresholdSignificantTimeInMs * 0.001f, timeThresholdMin);

        if (deviceTimingMode)
            timeThresholdMin = timeThresholdMax = 0;

        DisplayParams displayParams
        {
            timeThresholdFraction, timeThresholdMin, timeThresholdMax,
            timeMsDigits,
            powf(10.f, -float32(factorDigits)) / 2,
            factorDigits,
            maxSourceLineLength,
            maxCallstackFullyExpanded,
            minCallstackExpanded
        };

        HtmlReportParams params{timingParams, codeBlockParams, displayParams, outputDirPrefix, timeStr, sourceCacheThunk};
        generateHtmlForTree(*rootNode, rootInfo, StlString(), params, stdPass);

    }
    catch (exception& e)
    {
        printMsg(kit.msgLog, STR("STL exception: %0"), e.what(), msgErr);
    }
}

//----------------------------------------------------------------

}
