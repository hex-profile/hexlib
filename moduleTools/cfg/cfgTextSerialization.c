#include "cfgTextSerialization.h"


//================================================================
//
// CfgOutputStringSingleChar
//
//================================================================

class CfgOutputStringSingleChar : public CfgOutputString
{

public:

    virtual bool addBuf(const CharType* bufArray, size_t bufSize)
    {
        if (bufSize) value = *bufArray;
        charCount += bufSize;
        return true;
    }

    CharType value = ' ';
    size_t charCount = 0;

};

//================================================================
//
// CfgRead<bool>::func
//
//================================================================

bool CfgRead<bool>::func(CfgReadStream& s, bool& value)
{
    CfgOutputStringSingleChar result;
    ensure(s.readString(result));
    ensure(result.charCount >= size_t{1});

    ////

    CharType ch = result.value;

    bool detect0 = (ch == '0' || ch == 'N' || ch == 'n');
    bool detect1 = (ch == '1' || ch == 'Y' || ch == 'y');

    ensure(detect0 || detect1);

    ////

    value = detect1;
    return true;
}
