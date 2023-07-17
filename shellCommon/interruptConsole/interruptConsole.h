#pragma once

#include "interfaces/interrupt.h"
#include "storage/adapters/callable.h"

//================================================================
//
// InterruptSignaller
//
//================================================================

using InterruptSignaller = Callable<void ()>; // Totally async!

//================================================================
//
// InterruptConsole
//
//================================================================

class InterruptConsole : public Interrupt
{

public:

    virtual bool operator() ();

public:

    virtual void setSignaller(InterruptSignaller* signaller);

public:

    InterruptConsole();

};
