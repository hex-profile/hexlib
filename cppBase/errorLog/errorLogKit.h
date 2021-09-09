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

KIT_CREATE(ErrorLogKit, ErrorLog&, errorLog);
