#pragma once

#include "stl/stlArray.h"
#include "history/historyObj.h"

//================================================================
//
// StlHistory
//
//================================================================

template <typename Type>
using StlHistory = HistoryGeneric<StlArray<Type>>;
