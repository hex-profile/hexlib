#include "numericVar.inl"

//================================================================
//
// SerializeBoolVar::getTextValue
// SerializeBoolVar::setTextValue
//
//================================================================

bool SerializeBoolVar::getTextValue(CfgWriteStream& s) const
{
    ensure(cfgWrite(s, var != 0));
    return true;
}

//----------------------------------------------------------------

bool SerializeBoolVar::setTextValue(CfgReadStream& s) const
{
    bool readValue = false;
    ensure(cfgRead(s, readValue));
    var = readValue;
    return true;
}

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class SerializeNumericVar<Type>; \

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
