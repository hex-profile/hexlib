#pragma once

#include "prepTools/prepList.h"
#include "compileTools/compileTools.h"

//================================================================
//
// CLASS_CONTEXT
//
//================================================================

#define CLASS_CONTEXT(Class, seq) \
    CLASS_CONTEXT_EX2(Class, seq (_), PREP_EMPTY)

#define CLASS_CONTEXT_(Class, seq) \
    CLASS_CONTEXT_EX2(Class, seq (_), typename)

////

#define CLASS_CONTEXT_EX2(Class, list, typenameKeyword) \
    \
    public: \
        \
        inline Class(PREP_LIST_ENUM_PAIR(list, CLASS_CONTEXT_ARGUMENT, typenameKeyword)) \
            : PREP_LIST_ENUM_PAIR(list, CLASS_CONTEXT_CONSTRUCT_FIELD, typenameKeyword) \
        { \
        } \
        \
    private: \
        \
        PREP_LIST_FOREACH_PAIR(list, CLASS_CONTEXT_DECL_FIELD, typenameKeyword) \

#define CLASS_CONTEXT_ARGUMENT(Type, name, typenameKeyword) \
    typenameKeyword ParamType< Type >::T name

#define CLASS_CONTEXT_CONSTRUCT_FIELD(Type, name, typenameKeyword) \
    name(name)

#define CLASS_CONTEXT_DECL_FIELD(Type, name, typenameKeyword) \
    Type name;
