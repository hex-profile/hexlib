#pragma once

#include "connectionInterface.h"

#if defined(_WIN32)
    #include "connectionWin32.h"
    using ConnectionImpl = ConnectionWin32;
#endif
