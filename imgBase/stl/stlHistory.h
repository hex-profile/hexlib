#pragma once

#include "stl/stlArray.h"
#include "history/historyObject.h"

//================================================================
//
// StlHistory
//
//================================================================

template <typename Type>
using StlHistory = HistoryGeneric<StlArray<Type>>;
