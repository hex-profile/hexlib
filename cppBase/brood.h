#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// BroodParentRole
//
//================================================================

class BroodParentRole;

//================================================================
//
// BroodChildRole
//
//================================================================

class BroodChildRole
{

public:

    sysinline BroodChildRole()
    {
        parent = 0;
    }

public:

    sysinline BroodChildRole(const BroodChildRole& that);
    sysinline BroodChildRole& operator =(const BroodChildRole& that);

public:

    sysinline ~BroodChildRole()
        {disconnect();}

public:

    sysinline BroodParentRole* getParent() const
        {return parent;}

public:

    sysinline void disconnect();

private:

    BroodParentRole* parent;
    BroodChildRole* prev;
    BroodChildRole* next;

    friend class BroodParentRole;

};

//================================================================
//
// BroodParentRole
//
//================================================================

class BroodParentRole
{

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline BroodParentRole()
    {
        entry = 0;
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline ~BroodParentRole()
    {
        BroodChildRole* p = entry;

        if (p != 0)
        {
            do
            {
                BroodChildRole* next = p->next;

                p->parent = 0;
                p->prev = 0;
                p->next = 0;

                p = next;
            }
            while (p != entry);
        }

        ////

        entry = 0;
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline void connect(BroodChildRole& X)
    {

        if (X.parent)
            X.parent->disconnect(X);

        ////

        X.parent = this;

        if (entry == 0)
        {
            X.next = &X;
            X.prev = &X;

            entry = &X;
        }
        else
        {
            BroodChildRole& A = *entry;
            BroodChildRole& B = *entry->next;

            X.prev = &A;
            X.next = &B;

            A.next = &X;
            B.prev = &X;
        }

    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline bool connected(const BroodChildRole& X) const
        {return (X.parent == this);}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    static sysinline void disconnect(BroodChildRole& X)
    {
        BroodParentRole* that = X.parent;

        if (that != 0)
        {
            BroodChildRole& A = *X.prev;
            BroodChildRole& B = *X.next;

            A.next = &B;
            B.prev = &A;

            ////

            X.parent = 0;

            if (that->entry == &X)
                that->entry = &B;

            if (that->entry == &X)
                that->entry = 0;
        }

    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline BroodChildRole* getFirst()
    {
        return entry;
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    sysinline BroodChildRole* getNext(BroodChildRole* iter)
    {
        if (iter != 0)
        {
            iter = iter->next;

            if (iter == entry)
                iter = 0;
        }

        return iter;
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    void disconnectAll()
    {
        for (BroodChildRole* p = getFirst(); p != 0; p = getNext(p))
            disconnect(*p);
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

private:

    BroodChildRole* entry;

};

//================================================================
//
// BroodChildRole::disconnect
//
//================================================================

sysinline void BroodChildRole::disconnect()
{
    BroodParentRole::disconnect(*this);
}

//================================================================
//
// BroodChildRole::BroodChildRole
// BroodChildRole& BroodChildRole::operator =
//
//================================================================

sysinline BroodChildRole::BroodChildRole(const BroodChildRole& that)
{
    parent = 0;

    if (that.parent)
        that.parent->connect(*this);
}

//----------------------------------------------------------------

sysinline BroodChildRole& BroodChildRole::operator =(const BroodChildRole& that)
{
    if (that.parent)
        that.parent->connect(*this);

    return *this;
}

//================================================================
//
// BROOD_STRUCT_BY_FIELD
//
//================================================================

#define BROOD_STRUCT_BY_FIELD(X, Struct, Field) \
    (* (Struct*) (((char*) (&X)) - ptrdiff_t(offsetof(Struct, Field))))
