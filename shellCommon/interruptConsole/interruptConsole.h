#pragma once

#include "interfaces/interrupt.h"

//================================================================
//
// InterruptConsole
//
//================================================================

class InterruptConsole : public Interrupt
{

public:

    bool operator() ();

public:

    InterruptConsole();

};
