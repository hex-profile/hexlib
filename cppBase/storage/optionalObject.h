#pragma once

#include <string.h>

#include "storage/opaqueStruct.h"
#include "storage/constructDestruct.h"

//================================================================
//
// OptionalObject
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
class OptionalObject
{

public:

    sysinline explicit operator bool () const 
        {return constructorCalled;}

    ////

    sysinline Type* getPtr()
        {return constructorCalled ? &memory.template recast<Type>() : nullptr;}

    sysinline const Type* getPtr() const
        {return constructorCalled ? &memory.template recast<Type>() : nullptr;}

    ////

    sysinline Type& operator*()
        {return *getPtr();}

    sysinline const Type& operator*() const
        {return *getPtr();}

    ////

    sysinline Type* operator()()
        {return getPtr();}

    sysinline const Type* operator()() const
        {return getPtr();}

    ////

    sysinline Type* operator ->()
        {return getPtr();}

    sysinline const Type* operator ->() const
        {return getPtr();}

public:

    sysinline OptionalObject()
    {
    }

    sysinline ~OptionalObject()
    {
        destroy();
    }

    sysinline OptionalObject(const OptionalObject<Type>& that)
    {
        if (that.constructorCalled)
        {
            constructCopy(memory.template recast<Type>(), *that);
            constructorCalled = true;
        }
    }

    sysinline auto& operator =(const OptionalObject<Type>& that)
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
    sysinline OptionalObject(const That& that)
    {
        constructParamsVariadic(* (Type*) &memory, that);
        constructorCalled = true;
    }

    template <typename... Params>
    sysinline void create(Params&&... params)
    {
        destroy();

        constructParamsVariadic(memory.template recast<Type>(), params...);
        constructorCalled = true;
    }

    template <typename... Params>
    sysinline void createOptional(bool exists, Params&&... params)
    {
        destroy();

        if (exists)
        {
            constructParamsVariadic(memory.template recast<Type>(), params...);
            constructorCalled = true;
        }
    }

    template <typename That>
    sysinline auto& operator =(const That& that)
    {
        create(that);
        return *this;
    }

    sysinline void destroy()
    {
        if (constructorCalled)
        {
            destruct(memory.template recast<Type>());
            constructorCalled = false;
        }
    }

    sysinline void cancelDestructor()
    {
        constructorCalled = false;
    }

    sysinline bool operator ==(const OptionalObject<Type>& that) const
    {
        ensure(this->constructorCalled == that.constructorCalled);

        if (this->constructorCalled)
            ensure(*this->getPtr() == *that.getPtr());

        return true;
    }

private:

    OpaqueStruct<sizeof(Type)> memory;

    bool constructorCalled = false;

};

//================================================================
//
// allv<OptionalObject>
//
//================================================================

template <typename Type>
sysinline bool allv(const OptionalObject<Type>& value)
    {return bool{value};}
