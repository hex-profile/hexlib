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

    inline operator bool () const 
        {return constructorCalled;}

    ////

    inline Type* getPtr()
        {return constructorCalled ? &memory.template recast<Type>() : nullptr;}

    inline const Type* getPtr() const
        {return constructorCalled ? &memory.template recast<Type>() : nullptr;}

    ////

    inline operator Type*()
        {return getPtr();}

    inline operator const Type*() const
        {return getPtr();}

    ////

    inline Type* operator()()
        {return getPtr();}

    inline const Type* operator()() const
        {return getPtr();}

    ////

    inline Type* operator ->()
        {return getPtr();}

    inline const Type* operator ->() const
        {return getPtr();}

public:

    inline DisposableObject()
    {
    }

    inline ~DisposableObject()
    {
        destroy();
    }

    inline DisposableObject(const DisposableObject<Type>& that)
    {
        if (that.constructorCalled)
        {
            constructCopy(memory.template recast<Type>(), *that);
            constructorCalled = true;
        }
    }

    inline auto& operator =(const DisposableObject<Type>& that)
    {
        if (this != &that)
        {
            destroy();

            if (that.constructorCalled)
            {
                constructCopy(memory.template recast<Type>(), *that);
                constructorCalled = true;
            }
        }

        return *this;
    }

    template <typename That>
    inline DisposableObject(const That& that)
    {
        constructParamsVariadic(* (Type*) &memory, that);
        constructorCalled = true;
    }

    template <typename... Params>
    inline void create(Params&&... params)
    {
        destroy();

        constructParamsVariadic(memory.template recast<Type>(), params...);
        constructorCalled = true;
    }

    template <typename... Params>
    inline void createOptional(bool exists, Params&&... params)
    {
        destroy();

        if (exists)
        {
            constructParamsVariadic(memory.template recast<Type>(), params...);
            constructorCalled = true;
        }
    }

    template <typename That>
    inline auto& operator =(const That& that)
    {
        destroy();

        constructParamsVariadic(memory.template recast<Type>(), that);
        constructorCalled = true;

        return *this;
    }

    inline void destroy()
    {
        if (constructorCalled)
        {
            destruct(memory.template recast<Type>());
            constructorCalled = false;
        }
    }

    inline void cancelDestructor()
    {
        constructorCalled = false;
    }

private:

    OpaqueStruct<sizeof(Type)> memory;

    bool constructorCalled = false;

};
