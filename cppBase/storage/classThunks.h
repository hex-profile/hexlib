#pragma once

#include "prepTools/prepBase.h"
#include "storage/constructDestruct.h"

//================================================================
//
// Macros for generating class function thunks in
// isolated class implementation (StaticClass, etc).
//
//================================================================

//================================================================
//
// CLASSTHUNK_CONSTRUCT_DESTRUCT
//
// Thunks for constructor and destructor.
//
//================================================================

#define CLASSTHUNK_CONSTRUCT_DESTRUCT(Class) \
    \
    Class::Class() \
    { \
    } \
    \
    Class::~Class() \
    { \
    }

//================================================================
//
// CLASSTHUNK_PURE_EX*
//
// Thunks for class functions without standard parameters.
//
//================================================================

#define CLASSTHUNK_PURE0(Class, RetType, defaultRetVal, func, funcModifier) \
    RetType Class::func() funcModifier \
        {return instance->func();}

#define CLASSTHUNK_PURE1(Class, RetType, defaultRetVal, func, funcModifier, Type0) \
    RetType Class::func(Type0 v0) funcModifier \
        {return instance->func(v0);}

#define CLASSTHUNK_PURE2(Class, RetType, defaultRetVal, func, funcModifier, Type0, Type1) \
    RetType Class::func(Type0 v0, Type1 v1) funcModifier \
        {return instance->func(v0, v1);}

//----------------------------------------------------------------

#define CLASSTHUNK_VOID0(Class, func) \
    CLASSTHUNK_PURE0(Class, void, void(), func, PREP_EMPTY)

#define CLASSTHUNK_VOID1(Class, func, Type0) \
    CLASSTHUNK_PURE1(Class, void, void(), func, PREP_EMPTY, Type0)

#define CLASSTHUNK_VOID2(Class, func, Type0, Type1) \
    CLASSTHUNK_PURE2(Class, void, void(), func, PREP_EMPTY, Type0, Type1)

//----------------------------------------------------------------

#define CLASSTHUNK_BOOL0(Class, func) \
    CLASSTHUNK_PURE0(Class, bool, false, func, PREP_EMPTY)

#define CLASSTHUNK_BOOL1(Class, func, Type0) \
    CLASSTHUNK_PURE1(Class, bool, false, func, PREP_EMPTY, Type0)

#define CLASSTHUNK_BOOL2(Class, func, Type0, Type1) \
    CLASSTHUNK_PURE2(Class, bool, false, func, PREP_EMPTY, Type0, Type1)

//----------------------------------------------------------------

#define CLASSTHUNK_BOOL_CONST0(Class, func) \
    CLASSTHUNK_PURE0(Class, bool, false, func, const)

#define CLASSTHUNK_BOOL_CONST1(Class, func, Type0) \
    CLASSTHUNK_PURE1(Class, bool, false, func, const, Type0)

#define CLASSTHUNK_BOOL_CONST2(Class, func, Type0, Type1) \
    CLASSTHUNK_PURE2(Class, bool, false, func, const, Type0, Type1)

//================================================================
//
// CLASSTHUNK_STD*
//
// Thunks for class functions with standard parameters.
//
//================================================================

#define CLASSTHUNK_STDEX0(Class, RetType, defaultRetVal, func, funcModifier, Kit) \
    RetType Class::func(stdPars(Kit)) funcModifier \
        {return instance->func(stdPassThru);}

#define CLASSTHUNK_STDEX1(Class, RetType, defaultRetVal, func, funcModifier, Type0, Kit) \
    RetType Class::func(Type0 v0, stdPars(Kit)) funcModifier \
        {return instance->func(v0, stdPassThru);}

#define CLASSTHUNK_STDEX2(Class, RetType, defaultRetVal, func, funcModifier, Type0, Type1, Kit) \
    RetType Class::func(Type0 v0, Type1 v1, stdPars(Kit)) funcModifier \
        {return instance->func(v0, v1, stdPassThru);}

#define CLASSTHUNK_STDEX3(Class, RetType, defaultRetVal, func, funcModifier, Type0, Type1, Type2, Kit) \
    RetType Class::func(Type0 v0, Type1 v1, Type2 v2, stdPars(Kit)) funcModifier \
        {return instance->func(v0, v1, v2, stdPassThru);}

#define CLASSTHUNK_STDEX4(Class, RetType, defaultRetVal, func, funcModifier, Type0, Type1, Type2, Type3, Kit) \
    RetType Class::func(Type0 v0, Type1 v1, Type2 v2, Type3 v3, stdPars(Kit)) funcModifier \
        {return instance->func(v0, v1, v2, v3, stdPassThru);}

#define CLASSTHUNK_STDEX5(Class, RetType, defaultRetVal, func, funcModifier, Type0, Type1, Type2, Type3, Type4, Kit) \
    RetType Class::func(Type0 v0, Type1 v1, Type2 v2, Type3 v3, Type4 v4, stdPars(Kit)) funcModifier \
        {return instance->func(v0, v1, v2, v3, v4, stdPassThru);}

//----------------------------------------------------------------

#define CLASSTHUNK_BOOL_STD0(Class, func, Kit) \
    CLASSTHUNK_STDEX0(Class, stdbool, false, func, PREP_EMPTY, Kit)

#define CLASSTHUNK_BOOL_STD1(Class, func, Type0, Kit) \
    CLASSTHUNK_STDEX1(Class, stdbool, false, func, PREP_EMPTY, Type0, Kit)

#define CLASSTHUNK_BOOL_STD2(Class, func, Type0, Type1, Kit) \
    CLASSTHUNK_STDEX2(Class, stdbool, false, func, PREP_EMPTY, Type0, Type1, Kit)

#define CLASSTHUNK_BOOL_STD3(Class, func, Type0, Type1, Type2, Kit) \
    CLASSTHUNK_STDEX3(Class, stdbool, false, func, PREP_EMPTY, Type0, Type1, Type2, Kit)

#define CLASSTHUNK_BOOL_STD4(Class, func, Type0, Type1, Type2, Type3, Kit) \
    CLASSTHUNK_STDEX4(Class, stdbool, false, func, PREP_EMPTY, Type0, Type1, Type2, Type3, Kit)

#define CLASSTHUNK_BOOL_STD5(Class, func, Type0, Type1, Type2, Type3, Type4, Kit) \
    CLASSTHUNK_STDEX5(Class, stdbool, false, func, PREP_EMPTY, Type0, Type1, Type2, Type3, Type4, Kit)

//----------------------------------------------------------------

#define CLASSTHUNK_VOID_STD0(Class, func, Kit) \
    CLASSTHUNK_STDEX0(Class, void, void(), func, PREP_EMPTY, Kit)

#define CLASSTHUNK_VOID_STD1(Class, func, Type0, Kit) \
    CLASSTHUNK_STDEX1(Class, void, void(), func, Type0, Kit)

#define CLASSTHUNK_VOID_STD2(Class, func, Type0, Type1, Kit) \
    CLASSTHUNK_STDEX2(Class, void, void(), func, Type0, Type1, Kit)
