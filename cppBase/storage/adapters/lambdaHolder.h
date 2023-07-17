#pragma once

#include <type_traits>

#include "storage/opaqueStruct.h"

//================================================================
//
// LambdaHolder
//
// Contains a generic lambda of the specified signature.
//
// Efficient:
// * Uses static memory storage up to the specified number of bytes.
// * Supports only lambdas with empty destructor.
//
//================================================================

template <typename Proto, size_t maxSize>
class LambdaHolderBase;

////

template <typename ReturnType, typename... Arguments, size_t maxSize>
class LambdaHolderBase<ReturnType (Arguments...), maxSize>
{

public:

    //----------------------------------------------------------------
    //
    // Create / destroy.
    //
    //----------------------------------------------------------------

	sysinline LambdaHolderBase() =default;

    ////

	sysinline LambdaHolderBase(const LambdaHolderBase& that) =default;

	sysinline LambdaHolderBase& operator =(const LambdaHolderBase& that) =default;

    ////

	template <typename Lambda>
	sysinline LambdaHolderBase(const Lambda& lambda)
        {assignFunc(lambda);}

    ////

	template <typename Lambda>
	sysinline void operator =(const Lambda& lambda)
        {assignFunc(lambda);}

    ////

    template <typename Lambda>
    sysinline void assignFunc(const Lambda& lambda)
    {
        COMPILE_ASSERT(sizeof(HolderImpl<Lambda>) <= maxSize);

        COMPILE_ASSERT(std::is_trivially_destructible<Lambda>::value);
        COMPILE_ASSERT(std::is_nothrow_destructible<Lambda>::value);
        COMPILE_ASSERT(std::is_nothrow_copy_constructible<Lambda>::value);

        auto& impl = data.recast<HolderImpl<Lambda>>();
        constructParamsVariadic(impl, lambda);

        holder = &impl;
    }

    ////

    sysinline auto& operator= (std::nullptr_t)
    {
        holder = nullptr;
        return *this;
    }

    //----------------------------------------------------------------
    //
    // Check.
    //
    //----------------------------------------------------------------

    sysinline explicit operator bool() const
    {
        return holder != 0;
    }

    //----------------------------------------------------------------
    //
    // Call.
    //
    //----------------------------------------------------------------

	sysinline auto operator ()(Arguments... args) const
    {
		return holder->invoke(args...);
	}

    //----------------------------------------------------------------
    //
    // Holder.
    //
    //----------------------------------------------------------------

private:

	struct Holder
    {
		virtual ReturnType invoke(Arguments... args) const =0;
	};

    ////

	template <typename Lambda>
	struct HolderImpl : public Holder
    {
		sysinline HolderImpl(const Lambda& lambda)
            : lambda(lambda) {}

		virtual ReturnType invoke(Arguments... args) const
            {return lambda(args...);}

		Lambda lambda;
	};

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    Holder* holder = nullptr;
    OpaqueStruct<maxSize, 0> data;

};

////

template <typename Proto, size_t maxSize = 16>
using LambdaHolder = LambdaHolderBase<Proto, maxSize>;
