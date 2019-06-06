#pragma once

#include "data/space.h"

//================================================================
//
// History
//
// History access interface
//
//================================================================

template <typename Element>
struct History
{
    virtual Element* operator [] (Space index) =0;
};

//================================================================
//
// HistoryThunk
//
// Adapter of history containers to history interface.
//
//================================================================

template <typename History>
class HistoryThunk
{

public:

    auto operator [] (Space index)
        {return history[index];}

    inline HistoryThunk(History& history)
        : history(history) {}

private:

    History& history;

};

//----------------------------------------------------------------

template <typename History>
inline HistoryThunk<History> historyThunk(History& history)
    {return HistoryThunk<History>(history);}
