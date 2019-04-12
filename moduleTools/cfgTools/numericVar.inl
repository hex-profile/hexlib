#include "numericVar.h"

#include "numbers/float/floatType.h"
#include "numbers/int/intType.h"
#include "point/point.h"

//================================================================
//
// SerializeNumericVar<Type>::getName
//
//================================================================

template <typename Type>
bool SerializeNumericVar<Type>::getName(CfgOutputString& getName) const
{
    return getName.addStr(name);
}

//================================================================
//
// SerializeNumericVar<Type>::getTextValue
//
//================================================================

template <typename Type>
bool SerializeNumericVar<Type>::getTextValue(CfgWriteStream& s) const
{
    require(cfgWrite(s, targetVar()));
    return true;
}

//================================================================
//
// SerializeNumericVar<Type>::setTextValue
//
//================================================================

template <typename Type>
bool SerializeNumericVar<Type>::setTextValue(CfgReadStream& s) const 
{
    Type readValue = targetVar.defaultValue();
    require(cfgRead(s, readValue));

    targetVar = readValue;
    return true;
}

//================================================================
//
// SerializeNumericVar<Type>::getTextComment
//
//================================================================

template <typename Type>
bool SerializeNumericVar<Type>::getTextComment(CfgWriteStream& s) const
{
    if_not (comment.size == 1 && *comment.ptr == '*')
        require(cfgWrite(s, comment));
    else
    {
        require(cfgWrite(s, targetVar.minValue()));
        require(cfgWrite(s, STR(" .. (")));
        require(cfgWrite(s, targetVar.defaultValue()));
        require(cfgWrite(s, STR(") .. ")));
        require(cfgWrite(s, targetVar.maxValue()));
    }

    return true;
}

//================================================================
//
// SerializeNumericVar<Type>::getBlockComment
//
//================================================================

template <typename Type>
bool SerializeNumericVar<Type>::getBlockComment(CfgWriteStream& s) const
{
    cfgWrite(s, blockComment);
    return true;
}
