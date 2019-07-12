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
    ensure(cfgWrite(s, targetVar()));
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
    Type readValue;
    ensure(cfgRead(s, readValue));
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
        ensure(cfgWrite(s, comment));
    else
    {
        ensure(cfgWrite(s, targetVar.minValue()));
        ensure(cfgWrite(s, STR(" .. (")));
        ensure(cfgWrite(s, targetVar.defaultValue()));
        ensure(cfgWrite(s, STR(") .. ")));
        ensure(cfgWrite(s, targetVar.maxValue()));
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
    ensure(cfgWrite(s, blockComment));
    return true;
}
