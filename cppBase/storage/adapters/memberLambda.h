#pragma once

//================================================================
//
// MemberLambda
//
// * Makes a virtual interface with a single function
// of the specified signature.
//
// * Provides an adapter from a lambda to the interface.
//
//================================================================

template <typename Object, typename Function>
class MemberLambda;

//----------------------------------------------------------------

template <typename Object, typename ReturnType, typename... Arguments>
class MemberLambda<Object, ReturnType (Object::*) (Arguments...)>
{

    //----------------------------------------------------------------
    //
    // Construct.
    //
    //----------------------------------------------------------------

public:

    using Function = ReturnType (Object::*) (Arguments...);

    inline MemberLambda(Object& object, Function function)
        : object{object}, function{function}
    {
    }

    //----------------------------------------------------------------
    //
    // Call.
    //
    //----------------------------------------------------------------

	inline ReturnType operator ()(Arguments... args) const
    {
        return (object.*function)(args...);
    }

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    Object& object;
    Function function;

};

////

template <typename Object, typename Function>
inline MemberLambda<Object, Function> memberLambda(Object& object, Function function)
{
    return {object, function};
}
