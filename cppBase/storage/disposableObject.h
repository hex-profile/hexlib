#pragma once

#include <string.h>

#include "storage/opaqueStruct.h"
#include "storage/constructDestruct.h"

//================================================================
//
// DisposableObject
//
// Wrapper of a class object's memory which can have null state
// (the object is not constructed) and filled state (the object is constructed).
//
// The tool supports explicit transitions between null and filled state.
//
// When the object is in null state, cast operators and -> operator
// return NULL pointer.
//
//================================================================

template <typename Type>
class DisposableObject
{

public:

    inline operator Type* ()
        {return constructorCalled ? (Type*) (&rawMemory) : 0;}

    inline operator const Type* () const
        {return constructorCalled ? (const Type*) (&rawMemory) : 0;}

    inline Type* operator()()
        {return constructorCalled ? (Type*) (&rawMemory) : 0;}

    inline const Type* operator()() const
        {return constructorCalled ? (const Type*) (&rawMemory) : 0;}

    inline Type* operator ->()
        {return constructorCalled ? (Type*) (&rawMemory) : 0;}

    inline const Type* operator ->() const
        {return constructorCalled ? (const Type*) (&rawMemory) : 0;}

public:

    inline DisposableObject()
    {
    }

    inline ~DisposableObject()
    {
        destroy();
    }

    template <typename... Params>
    inline void create(Params&&... params)
    {
        destroy();

        constructParamsVariadic(* (Type*) &rawMemory, params...);
        constructorCalled = true;
    }

    template <typename... Params>
    inline DisposableObject<Type>& operator =(Params&&... params)
    {
        destroy();

        constructParamsVariadic(* (Type*) &rawMemory, params...);
        constructorCalled = true;

        return *this;
    }

    inline void destroy()
    {
        if (constructorCalled)
        {
            destruct(* (Type*) (&rawMemory));
            constructorCalled = false;
        }
    }

    inline void cancelDestructor()
    {
        constructorCalled = false;
    }

    inline bool constructed() const
    {
        return constructorCalled;
    }

private:

    bool constructorCalled = false;

    OpaqueStruct<sizeof(Type)> rawMemory;

};
