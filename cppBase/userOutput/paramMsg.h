#pragma once

#include "formatting/paramMsgOutput.h"
#include "prepTools/prepEnum.h"
#include "formatting/formatModifiers.h"
#include "prepTools/prepFor.h"

//================================================================
//
// paramMsg
//
// Usage:
//
// paramMsg(format, arg0, ..., argN)
//
//================================================================

#define PARAMMSG__MAX_COUNT 16

//----------------------------------------------------------------

class ParamMsg0 : public ParamMsg
{

public:

    sysinline ParamMsg0(const CharArray& format)
        : ParamMsg(format, 0, 0) {}

};

sysinline ParamMsg0 paramMsg(const CharArray& format)
    {return ParamMsg0(format);}

//----------------------------------------------------------------

#define PARAMMSG__DECL_VALUE(k, o) \
    const T##k v##k;

#define PARAMMSG__COPY_VALUE(k, o) \
    v##k(v##k)

#define PARAMMSG__SET_ATOM(k, o) \
    params[k].setup(v##k);

//----------------------------------------------------------------

#define PARAMMSG__PRINT_MSG(ParamStruct, n) \
    \
    template <PREP_ENUM_INDEXED(n, typename T)> \
    class ParamStruct : public ParamMsg \
    { \
        \
    public: \
        \
        explicit sysinline ParamStruct \
        ( \
            const CharArray& format, \
            PREP_ENUM_INDEXED_PAIR(n, const T, &v) \
        ) \
            : \
            ParamMsg(format, params, n), \
            PREP_ENUM(n, PARAMMSG__COPY_VALUE, o) \
        { \
            PREP_FOR(n, PARAMMSG__SET_ATOM, o) \
        } \
        \
    private: \
        \
        PREP_FOR(n, PARAMMSG__DECL_VALUE, o) \
        FormatOutputAtom params[n]; \
        \
    }; \
    \
    template <PREP_ENUM_INDEXED(n, typename T)> \
    sysinline ParamStruct<PREP_ENUM_INDEXED(n, T)> paramMsg \
    ( \
        const CharArray& format, \
        PREP_ENUM_INDEXED_PAIR(n, const T, &v) \
    ) \
    { \
        return ParamStruct<PREP_ENUM_INDEXED(n, T)>(format, PREP_ENUM_INDEXED(n, v)); \
    } \
    \
    template <PREP_ENUM_INDEXED(n, typename T)> \
    sysinline ParamStruct<PREP_ENUM_INDEXED(n, T)> paramMsgSafe \
    ( \
        const CharArray& format, \
        PREP_ENUM_INDEXED_PAIR(n, const T, *v) \
    ) \
    { \
        return ParamStruct<PREP_ENUM_INDEXED(n, T)>(format, PREP_ENUM_INDEXED(n, *v)); \
    } \
    \
    template <PREP_ENUM_INDEXED(n, typename T)> \
    struct FormatOutputFunc<ParamStruct<PREP_ENUM_INDEXED(n, T)>> \
    { \
        typedef void FuncType(const ParamMsg& value, FormatOutputStream& outputStream); \
        static sysinline FuncType* get() {return &formatOutput<ParamMsg>;} \
    };

#define PARAMMSG__PRINT_MSG_THUNK(n, o) \
    PARAMMSG__PRINT_MSG(PREP_PASTE2(ParamMsg, PREP_INC(n)), PREP_INC(n))

PREP_FOR1(PARAMMSG__MAX_COUNT, PARAMMSG__PRINT_MSG_THUNK, o)
