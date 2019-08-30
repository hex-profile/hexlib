#pragma once

#include "cfg/cfgInterface.h"
#include "numbers/interface/numberInterface.h"
#include "numbers/int/intType.h"
#include "numbers/float/floatType.h"

//================================================================
//
// NumericVar
//
// Variable with limited range (min/default/max are specified)
// and tracking of value changes.
//
// Can be used as config var for ints/floats.
//
// Use SerializeNumericVar for serialization.
//
//================================================================

template <typename Type>
class NumericVar
{

public:

    using TypePar = const Type&;

public:

    using Self = NumericVar<Type>;

private:

    Type value;

    Type minVal;
    Type maxVal;
    Type defaultVal;

public:

    bool changed = false;

public:

    inline NumericVar(TypePar minVal, TypePar maxVal, TypePar defaultVal)
    {
        setup(minVal, maxVal, defaultVal);
    }

    inline NumericVar()
    {
        Type zero = convertNearest<Type>(0);
        setup(zero, zero, zero);
    }

    inline void setup(TypePar minVal, TypePar maxVal, TypePar defaultVal)
    {
        Type fixMinVal = minVal;
        Type fixMaxVal = clampMin(maxVal, minVal);
        Type fixDefaultVal = clampRange(defaultVal, fixMinVal, fixMaxVal);

        this->minVal = fixMinVal;
        this->maxVal = fixMaxVal;
        this->defaultVal = fixDefaultVal;
        this->value = fixDefaultVal;
    }

public:

    inline Self& operator =(TypePar X)
    {
        if_not (def(X))
            return *this;

        Type newValue = clampRange(X, minVal, maxVal);

        if_not (allv(newValue == value))
        {
            value = newValue;
            changed = true;
        }

        return *this;
    }

public:

    inline const Type& minValue() const
        {return minVal;}

    inline const Type& maxValue() const
        {return maxVal;}

    inline const Type& defaultValue() const
        {return defaultVal;}

public:

    inline operator const Type& () const
        {return value;}

    inline const Type& operator()() const
        {return value;}

public:

    #define TMP_MACRO(OP) \
        template <typename AnyType> \
        inline friend bool operator OP(const Self& A, const AnyType& B) \
            {return A.value OP B;}

    TMP_MACRO(==)
    TMP_MACRO(!=)
    TMP_MACRO(>)
    TMP_MACRO(<)
    TMP_MACRO(>=)
    TMP_MACRO(<=)

    #undef TMP_MACRO

public:

    //
    // Returns true if the variable was NOT changed.
    //

    inline bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& comment = STR(""), const CharArray& blockComment = STR(""));

};

//================================================================
//
// NumericVarStatic
//
// The same as NumericVar, but min/default/max are specified
// as template parameters, for convenience.
//
//================================================================

template <typename Type, Type compileMinVal, Type compileMaxVal, Type compileDefaultVal>
class NumericVarStatic : public NumericVar<Type>
{

    COMPILE_ASSERT(compileMinVal <= compileDefaultVal && compileDefaultVal <= compileMaxVal);

public:

    inline NumericVarStatic()
        : NumericVar<Type>(compileMinVal, compileMaxVal, compileDefaultVal) {}

    inline NumericVarStatic& operator =(const Type& X)
        {NumericVar<Type>::operator =(X); return *this;}

};

//----------------------------------------------------------------

template <typename Type, typename ConstantType, ConstantType compileMinVal, ConstantType compileMaxVal, ConstantType compileDefaultVal>
class NumericVarStaticEx : public NumericVar<Type>
{

    COMPILE_ASSERT(compileMinVal <= compileDefaultVal && compileDefaultVal <= compileMaxVal);

public:

    inline NumericVarStaticEx()
        :
        NumericVar<Type>
        (
            convertNearest<Type>(compileMinVal),
            convertNearest<Type>(compileMaxVal),
            convertNearest<Type>(compileDefaultVal)
        )
    {
    }

    inline NumericVarStaticEx(const Type& value)
        :
        NumericVarStaticEx()
    {
        NumericVar<Type>::operator =(value);
    }

    inline NumericVarStaticEx& operator =(const Type& X)
    {
        NumericVar<Type>::operator =(X);
        return *this;
    }

};

//================================================================
//
// SerializeNumericVar
//
// Serializes NumericVar.
// If comment is '*', generates automatic comment about min/max/default.
//
//================================================================

template <typename Type>
class SerializeNumericVar : public CfgSerializeVariable
{

private:

    NumericVar<Type>& targetVar;
    const CharArray name;
    const CharArray comment;
    const CharArray blockComment;

public:

    inline SerializeNumericVar(NumericVar<Type>& targetVar, const CharArray& name, const CharArray& comment, const CharArray& blockComment)
        :
        targetVar(targetVar),
        name(name),
        comment(comment),
        blockComment(blockComment)
    {
    }

    bool changed() const 
        {return targetVar.changed;}

    void clearChanged() const 
        {targetVar.changed = false;}

    void resetValue() const
        {targetVar = targetVar.defaultValue();}

public:

    bool getName(CfgOutputString& getName) const;
    bool getTextComment(CfgWriteStream& s) const;
    bool getBlockComment(CfgWriteStream& s) const;
    bool getTextValue(CfgWriteStream& s) const;
    bool setTextValue(CfgReadStream& s) const;

};

//================================================================
//
// NumericVar<Type>::serialize
//
//================================================================

template <typename Type>
inline bool NumericVar<Type>::serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& comment, const CharArray& blockComment)
{
    Type oldValue = value;
    SerializeNumericVar<Type> serializeVar(*this, name, comment, blockComment);
    kit.visitor(kit.scope, serializeVar);
    return allv(oldValue == value);
}

//================================================================
//
// SerializeBoolVar
//
//================================================================

class SerializeBoolVar : public SerializeNumericVar<int32>
{

public:

    inline SerializeBoolVar(NumericVar<int32>& var, const CharArray& name, const CharArray& comment, const CharArray& blockComment)
        : SerializeNumericVar<int32>(var, name, comment, blockComment), var(var) {}

    bool getTextValue(CfgWriteStream& s) const;
    bool setTextValue(CfgReadStream& s);

private:

    NumericVar<int32>& var;

};

//================================================================
//
// BoolVarStatic
//
//================================================================

class BoolVar : public NumericVar<int32>
{

    using Base = NumericVar<int32>;

public:

    inline operator bool () const {return Base::operator()() != 0;}

    inline BoolVar(bool value)
        : Base(0, 1, int32(value))
    {
    }

    inline BoolVar& operator=(bool value)
    {
        Base::operator=(int32(value));
        return *this;
    }

    inline bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& comment = STR(""), const CharArray& blockComment = STR(""))
    {
        int32 oldValue = Base::operator()();

        SerializeBoolVar serializeVar(*this, name, comment, blockComment);
        kit.visitor(kit.scope, serializeVar);

        return Base::operator()() == oldValue;
    }

};

//================================================================
//
// BoolVarStatic
//
//================================================================

template <bool defaultBool>
class BoolVarStatic : public NumericVarStatic<int32, 0, 1, defaultBool>
{

    using Base = NumericVarStatic<int32, 0, 1, defaultBool>;

public:

    inline operator bool () const {return Base::operator()() != 0;}

    inline BoolVarStatic() =default;

    inline BoolVarStatic(bool value)
    {
        Base::operator=(int32(value));
    }

    inline BoolVarStatic<defaultBool>& operator=(bool value)
    {
        Base::operator=(int32(value));
        return *this;
    }

    inline bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& comment = STR(""), const CharArray& blockComment = STR(""))
    {
        int32 oldValue = Base::operator()();

        SerializeBoolVar serializeVar(*this, name, comment, blockComment);
        kit.visitor(kit.scope, serializeVar);

        return Base::operator()() == oldValue;
    }

};
