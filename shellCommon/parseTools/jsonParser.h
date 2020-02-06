#pragma once

namespace jsonParser {

//================================================================
//
// Range
//
//================================================================

template <typename Iterator>
struct Range
{
    Iterator ptr;
    Iterator end;
};

//----------------------------------------------------------------

template <typename Iterator>
constexpr inline auto makeRange(Iterator ptr, Iterator end)
    {return Range<Iterator>{ptr, end};}

template <typename StaticArray>
constexpr inline auto makeRangeFromArray(const StaticArray& staticArray)
    {return makeRange(staticArray, staticArray + COMPILE_ARRAY_SIZE(staticArray));}

//================================================================
//
// equal
//
//================================================================

template <typename Iterator>
inline bool operator ==(const Range<Iterator>& A, const Range<Iterator>& B);

//================================================================
//
// Index
//
//================================================================

using Index = unsigned;
static constexpr Index invalidIndex = unsigned(-1);

//================================================================
//
// Key
//
// Contains either a string or an index.
// If it's index, the string is empty.
// If it's string, the index is invalid index.
//
//================================================================

template <typename Iterator>
struct Key
{
    Range<Iterator> name;
    Index index;
};

//----------------------------------------------------------------

template <typename Iterator>
inline bool isIndex(const Key<Iterator>& key)
{
    return key.index != invalidIndex;
}

//----------------------------------------------------------------

template <typename Iterator>
constexpr inline auto makeKey(const Range<Iterator>& name)
{
    return Key<Iterator>{name, invalidIndex};
}

template <typename Literal>
constexpr inline auto makeKey(const Literal& str)
{
    COMPILE_ASSERT(COMPILE_ARRAY_SIZE(str) >= 1);
    return makeKey(makeRange(str, str + COMPILE_ARRAY_SIZE(str) - 1));
}

//----------------------------------------------------------------

template <typename Iterator>
inline bool operator ==(const Key<Iterator>& A, const Key<Iterator>& B)
{
    return 
        A.index == B.index &&
        A.name == B.name;
}

//================================================================
//
// equal
//
//================================================================

template <typename Iterator>
inline bool operator ==(const Range<Iterator>& A, const Range<Iterator>& B)
{
    auto sizeA = A.end - A.ptr; 
    auto sizeB = B.end - B.ptr; 

    ensure(sizeA == sizeB);

    auto ptrA = A.ptr;
    auto ptrB = B.ptr;

    for (size_t i = 0; i != sizeA; ++i)
    {
        ensure(*ptrA == *ptrB);
        ++ptrA; ++ptrB;
    }

    return true;
}

//================================================================
//
// Visitor
//
//================================================================

template <typename Iterator>
struct Visitor
{               
    // Enter namespace and leave namespace. 
    // Enter function should control max depth and return false in case of error.
    virtual bool enter(const Key<Iterator>& key) =0;
    virtual void leave() =0;

    // Handle value at current location. This function can interrupt processing
    // by returning false, in this case parsing also returns false.
    virtual bool handleValue(const Range<Iterator>& value) =0;

    // Handle value at current location.
    virtual void handleError(Iterator errorPlace) =0;
};

//================================================================
//
// PathVisitor
//
//================================================================

template <typename Iterator>
struct PathVisitor
{
    virtual bool handleValue(const Range<const Key<Iterator>*>& path, const Range<Iterator>& value) =0;
    virtual void handleError(const Range<const Key<Iterator>*>& path, Iterator errorPlace) =0;
};

//================================================================
//
// PathVisitorImpl
//
//================================================================

template <typename Iterator, int maxDepth = 64>
class PathVisitorImpl : public Visitor<Iterator>
{

public:

    PathVisitorImpl(PathVisitor<Iterator>& base)
        : base(base) {}

    virtual bool enter(const Key<Iterator>& key)
    {
        if_not (depth < maxDepth)
            return false;

        path[depth++] = key;
        return true;
    }

    virtual void leave()
    {
        if (depth) 
            --depth;
    }

    virtual bool handleValue(const Range<Iterator>& value)
    {
        return base.handleValue({path, path + depth}, value);
    }

    virtual void handleError(Iterator errorPlace)
    {
        return base.handleError({path, path + depth}, errorPlace);
    }

private:

    Key<Iterator> path[maxDepth];
    int depth = 0;

    PathVisitor<Iterator>& base;

};

//================================================================
//
// parseJson
//
//================================================================

template <typename Iterator>
bool parseJson(const Range<Iterator>& json, Visitor<Iterator>& visitor);

//================================================================
//
// getRowCol
//
//================================================================

template <typename Iterator>
void getRowCol(const Range<Iterator>& buffer, Iterator place, int& resX, int& resY);

//----------------------------------------------------------------

}
