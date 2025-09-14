#pragma once

#include "dbgptr/dbgptrProtos.h"

//================================================================
//
// DbgptrReference
//
// The object's memory size and layout match "Pointer" argument exactly.
//
//================================================================

template <typename Pointer>
class DbgptrReference
{

public:

    using Element = typename Pointer::Element;

private:

    using Self = DbgptrReference<Pointer>;

public:

    DbgptrReference() =delete;

    DbgptrReference(const Self&) =default;

public:

    sysinline const Pointer& asPointer() const
        {return pointer;}

    sysinline Pointer& asPointer()
        {return pointer;}

public:

    sysinline const Element& read() const
        {return pointer.read();}

    template <typename Type>
    sysinline void write(const Type& value) const
        {return pointer.write(value);}

    sysinline Element& modify() const
        {return pointer.modify();}

    //
    // Assignment operator.
    //

public:

    sysinline const Self& operator =(const Self& that) const
        {this->write(that.read()); return *this;}

    template <typename AnyPointer>
    sysinline const Self& operator =(const DbgptrReference<AnyPointer>& that) const
        {this->write(that.read()); return *this;}

    template <typename Type>
    sysinline const Self& operator =(const Type& value) const
        {write(value); return *this;}

    //
    // Taking reference: &.
    // Cast to Element type.
    //

public:

    sysinline Pointer operator &() const
        {return pointer;}

    sysinline operator const Element& () const
        {return read();}

    sysinline operator Element& ()
        {return modify();}

    sysinline Element* operator -> ()
        {return &modify();}

private:

    Pointer pointer;

};

//================================================================
//
// DbgptrReference: Unary plus and minus.
//
//================================================================

template <typename Pointer>
sysinline auto operator +(const DbgptrReference<Pointer>& ref)
{
    return +ref.read();
}

template <typename Pointer>
sysinline auto operator -(const DbgptrReference<Pointer>& ref)
{
    return -ref.read();
}

//================================================================
//
// DbgptrReference: Binary operations.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename Pointer1, typename Pointer2> \
    sysinline auto operator OP \
    ( \
        const DbgptrReference<Pointer1>& v1, \
        const DbgptrReference<Pointer2>& v2 \
    ) \
    { \
        using Element1 = typename Pointer1::Element; \
        using Element2 = typename Pointer2::Element; \
        \
        return v1.read() OP v2.read(); \
    } \
    \
    template <typename Pointer, typename Value> \
    sysinline auto operator OP \
    ( \
        const DbgptrReference<Pointer>& v1, \
        const Value& v2 \
    ) \
    { \
        using Element1 = typename Pointer::Element; \
        using Element2 = Value; \
        \
        return v1.read() OP v2; \
    } \
    \
    template <typename Pointer, typename Value> \
    sysinline auto operator OP \
    ( \
        const Value& v1, \
        const DbgptrReference<Pointer>& v2 \
    ) \
    { \
        using Element1 = Value; \
        using Element2 = typename Pointer::Element; \
        \
        return v1 OP v2.read(); \
    }

//----------------------------------------------------------------

TMP_MACRO(+)
TMP_MACRO(-)
TMP_MACRO(*)
TMP_MACRO(/)
//TMP_MACRO(%)
//TMP_MACRO(>>)
//TMP_MACRO(<<)
//TMP_MACRO(|)
//TMP_MACRO(&)
//TMP_MACRO(^)

#undef TMP_MACRO

//================================================================
//
// DbgptrReference: Assignment arithmetic.
//
//================================================================

#define TMP_MACRO(ASGOP) \
    \
    template <typename Value, typename Pointer> \
    sysinline auto operator ASGOP(Value& L, const DbgptrReference<Pointer>& R) \
    { \
        return L ASGOP R.read(); \
    } \
    \
    template <typename Value, typename Pointer> \
    sysinline auto operator ASGOP(const DbgptrReference<Pointer>& L, const Value& R) \
    { \
        return L.modify() ASGOP R; \
    } \
    \
    template <typename Pointer1, typename Pointer2> \
    sysinline auto operator ASGOP(const DbgptrReference<Pointer1>& L, const DbgptrReference<Pointer2>& R) \
    { \
        return L.modify() ASGOP R.read(); \
    }

TMP_MACRO(+=)
TMP_MACRO(-=)
TMP_MACRO(*=)
TMP_MACRO(/=)
//TMP_MACRO(%=)
//TMP_MACRO(>>=)
//TMP_MACRO(<<=)
//TMP_MACRO(|=)
//TMP_MACRO(&=)
//TMP_MACRO(^=)

#undef TMP_MACRO

//================================================================
//
// DbgptrReference: Comparisons.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename PointerA, typename PointerB> \
    sysinline auto operator OP \
    ( \
        const DbgptrReference<PointerA>& A, \
        const DbgptrReference<PointerB>& B \
    ) \
    { \
        return A.read() OP B.read(); \
    } \
    \
    template <typename Value, typename Pointer> \
    sysinline auto operator OP \
    ( \
        const DbgptrReference<Pointer>& A, \
        const Value& B \
    ) \
    { \
        return A.read() OP B; \
    } \
    \
    template <typename Value, typename Pointer> \
    sysinline auto operator OP \
    ( \
        const Value& A, \
        const DbgptrReference<Pointer>& B \
    ) \
    { \
        return A OP B.read(); \
    }

TMP_MACRO(>)
TMP_MACRO(<)
TMP_MACRO(>=)
TMP_MACRO(<=)
TMP_MACRO(==)
TMP_MACRO(!=)

#undef TMP_MACRO

//================================================================
//
// DbgptrReference: ++ and --, prefix and postfix.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename Pointer> \
    sysinline auto operator OP(const DbgptrReference<Pointer>& ref) \
    { \
        return OP ref.modify(); \
    } \
    \
    template <typename Pointer> \
    sysinline auto operator OP(const DbgptrReference<Pointer>& ref, int) \
    { \
        return ref.modify() OP; \
    }

TMP_MACRO(++)
TMP_MACRO(--)

#undef TMP_MACRO

//================================================================
//
// helpRead
// helpModify
//
//================================================================

template <typename Pointer>
sysinline const typename Pointer::Element& helpRead(const DbgptrReference<Pointer>& ref)
    {return ref.read();}

////

template <typename Pointer>
sysinline typename Pointer::Element& helpModify(const DbgptrReference<Pointer>& ref)
    {return ref.modify();}

//================================================================
//
// DBGPTR_DEFINE_FUNC1
//
//================================================================

#define DBGPTR_DEFINE_FUNC1(func) \
    \
    template <typename Pointer> \
    sysinline auto func(const DbgptrReference<Pointer>& ref) \
    { \
        return func(ref.read()); \
    }

//================================================================
//
// DBGPTR_DEFINE_FUNC2
//
//================================================================

#define DBGPTR_DEFINE_FUNC2(func) \
    \
    template <typename Pointer> \
    sysinline auto func \
    ( \
        const DbgptrReference<Pointer>& A, \
        const DbgptrReference<Pointer>& B \
    ) \
    { \
        return func(A.read(), B.read()); \
    } \
    \
    template <typename Pointer> \
    sysinline auto func \
    ( \
        const DbgptrReference<Pointer>& A, \
        const typename Pointer::Element& B \
    ) \
    { \
        return func(A.read(), B); \
    } \
    \
    template <typename Pointer> \
    sysinline auto func \
    ( \
        const typename Pointer::Element& A, \
        const DbgptrReference<Pointer>& B \
    ) \
    { \
        return func(A, B.read()); \
    }

//================================================================
//
// absv
// def
// minv
// maxv
//
//================================================================

DBGPTR_DEFINE_FUNC1(absv)
DBGPTR_DEFINE_FUNC1(def)
DBGPTR_DEFINE_FUNC2(maxv)
DBGPTR_DEFINE_FUNC2(minv)
