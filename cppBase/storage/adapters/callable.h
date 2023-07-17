#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// Callable
//
// * Makes a virtual interface with a single function
// of the specified signature.
//
// * Provides an adapter from a lambda to the interface.
//
//================================================================

template <typename Proto>
class Callable;

//================================================================
//
// CallableThunkMaker
//
//================================================================

template <typename ReturnType, typename... Arguments>
struct CallableThunkMaker
{
    template <typename Lambda>
    sysinline auto operator |(const Lambda& lambda) const;
};

//================================================================
//
// Callable implementation.
//
//================================================================

template <typename ReturnType, typename... Arguments>
struct Callable<ReturnType (Arguments...)>
{

    //----------------------------------------------------------------
    //
    // Interface.
    //
    //----------------------------------------------------------------

	virtual ReturnType operator ()(Arguments... args) =0;

	virtual ReturnType call(Arguments... args)
        {return (*this)(args...);}

    //----------------------------------------------------------------
    //
    // Thunk maker.
    //
    //----------------------------------------------------------------

    static constexpr CallableThunkMaker<ReturnType, Arguments...> O = {};

};

//================================================================
//
// CallableThunk
//
//================================================================

template <typename Lambda, typename ReturnType, typename... Arguments>
struct CallableThunk : public Callable<ReturnType (Arguments...)>
{
	sysinline CallableThunk(const Lambda& lambda)
        : lambda(lambda) {}

	virtual ReturnType operator()(Arguments... args)
        {return lambda(args...);}

	Lambda lambda;
};

//================================================================
//
// CallableThunkMaker::operator |
//
//================================================================

template <typename ReturnType, typename... Arguments>
template <typename Lambda>
sysinline auto CallableThunkMaker<ReturnType, Arguments...>::operator |(const Lambda& lambda) const
{
    return CallableThunk<Lambda, ReturnType, Arguments...>(lambda);
}
