#include "formatting/formatOutputEnum.h"

#include "imageRead/interpType.h"
#include "imageRead/borderMode.h"

//----------------------------------------------------------------

FORMAT_OUTPUT_ENUM_SIMPLE(InterpType, (INTERP_NONE) (INTERP_LINEAR) (INTERP_CUBIC) (INTERP_CUBIC_BSPLINE))
FORMAT_OUTPUT_ENUM_SIMPLE(BorderMode, (BORDER_ZERO) (BORDER_CLAMP) (BORDER_MIRROR) (BORDER_WRAP))
