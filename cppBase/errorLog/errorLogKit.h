#pragma once

#include "kit/kit.h"

//================================================================
//
// ErrorLogKit
//
// Default error reporter.
//
//================================================================

struct ErrorLog;

KIT_CREATE1(ErrorLogKit, ErrorLog&, errorLog);
