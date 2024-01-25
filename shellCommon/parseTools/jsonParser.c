#include "jsonParser.h"

#include "charType/charArray.h"
#include "compileTools/compileTools.h"
#include "parseTools/parseTools.h"
#include "storage/rememberCleanup.h"

namespace jsonParser {

//================================================================
//
// skipJsonValueBody
//
//================================================================

template <typename Iterator>
bool skipJsonValueBody(Iterator& ptr, Iterator end)
{
    auto s = ptr;

    ////

    ensure
    (
        skipJsonStr(s, end) ||
        skipFloat(s, end, false) ||
        skipText(s, end, STR("true")) ||
        skipText(s, end, STR("false")) ||
        skipText(s, end, STR("null"))
    );

    ptr = s;
    return true;
}

//================================================================
//
// Kit
//
//================================================================

template <typename Iterator>
struct Kit
{
    Iterator end = nullptr;
    Visitor<Iterator>* visitor = nullptr;
};

//----------------------------------------------------------------

#define PARSE_BEG \
    auto s = paramPtr; \
    const auto end = kit.end

#define PARSE_END \
    paramPtr = s; \
    return true

//----------------------------------------------------------------

#define ENSURE(condition, place) \
    if (condition) ; else {kit.visitor->handleError(place); return false;}

//================================================================
//
// parseElement
//
//================================================================

template <typename Iterator>
bool parseElement(Iterator& paramPtr, const Kit<Iterator>& kit);

//================================================================
//
// parseMember
//
//================================================================

template <typename Iterator>
bool parseMember(Iterator& paramPtr, const Kit<Iterator>& kit)
{
    PARSE_BEG;

    ////

    skipAnySpace(s, end);

    ////

    auto keyStart = s;
    ENSURE(skipJsonStr(s, end), s);

    ENSURE(s >= keyStart + 2, keyStart);

    Key<Iterator> key;
    key.index = invalidIndex;
    key.name = {keyStart + 1, s - 1};
    ENSURE(kit.visitor->enter(key), keyStart);
    REMEMBER_CLEANUP(kit.visitor->leave());

    skipAnySpace(s, end);
    ENSURE(skipTextThenAnySpace(s, end, STR(":")), s);

    ////

    ensure(parseElement(s, kit));

    ////

    PARSE_END;
}

//================================================================
//
// parseElement
//
//================================================================

template <typename Iterator>
bool parseElement(Iterator& paramPtr, const Kit<Iterator>& kit)
{
    PARSE_BEG;

    ////

    skipAnySpace(s, end);

    ////

    if (skipTextThenAnySpace(s, end, STR("{")))
    {
        if_not (skipTextThenAnySpace(s, end, STR("}")))
        {
            for (; ;)
            {
                ensure(parseMember(s, kit));

                if (skipTextThenAnySpace(s, end, STR("}")))
                    break;

                ENSURE(skipTextThenAnySpace(s, end, STR(",")), s);
            }
        }
    }

    ////

    else if (skipTextThenAnySpace(s, end, STR("[")))
    {
        if_not (skipTextThenAnySpace(s, end, STR("]")))
        {
            for (Index currentIndex = 0; ; ++currentIndex)
            {
                Key<Iterator> key;
                key.index = currentIndex;
                key.name = {s, s};
                ENSURE(kit.visitor->enter(key), s);
                REMEMBER_CLEANUP(kit.visitor->leave());

                ensure(parseElement(s, kit));

                if (skipTextThenAnySpace(s, end, STR("]")))
                    break;

                ENSURE(skipTextThenAnySpace(s, end, STR(",")), s);
            }
        }
    }

    ////

    else
    {
        auto valueStart = s;
        ENSURE(skipJsonValueBody(s, end), valueStart);
        ENSURE(kit.visitor->handleValue({valueStart, s}), valueStart);
        skipAnySpace(s, end);
    }

    ////

    skipAnySpace(s, end);

    ////

    PARSE_END;
}

//================================================================
//
// parseJson
//
//================================================================

template <typename Iterator>
bool parseJson(const Range<Iterator>& json, Visitor<Iterator>& visitor)
{
    Kit<Iterator> kit;
    kit.end = json.end;
    kit.visitor = &visitor;

    auto ptr = json.ptr;
    ensure(parseElement(ptr, kit));

    ENSURE(ptr == json.end, ptr);
    return true;
}

//================================================================
//
// getRowCol
//
//================================================================

template <typename Iterator>
void getRowCol(const Range<Iterator>& buffer, Iterator place, int& resX, int& resY)
{
    auto p = buffer.ptr;
    auto end = buffer.end;

    resX = 0;
    resY = 0;

    for (int row = 0; ; ++row)
    {
        Iterator linePtr = 0;
        Iterator lineEnd = 0;

        if_not (getNextLine(p, end, linePtr, lineEnd))
            break;

        if (place >= linePtr && place < lineEnd)
            {resX = int(place - linePtr); resY = row; break;}
    }
}

//================================================================
//
// Instantiations.
//
//================================================================

#define TMP_MACRO(Iterator) \
    INSTANTIATE_FUNC_EX(parseJson<Iterator>, 0) \
    INSTANTIATE_FUNC_EX(getRowCol<Iterator>, 1)

TMP_MACRO(char*)
TMP_MACRO(const char*)
TMP_MACRO(wchar_t*)
TMP_MACRO(const wchar_t*)

#undef TMP_MACRO

//----------------------------------------------------------------

}
