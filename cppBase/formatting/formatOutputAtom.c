#include "formatOutputAtom.h"

//================================================================
//
// formatOutput<>
//
//================================================================

template <>
void formatOutput(const FormatOutputAtom& atom, FormatOutputStream& outputStream)
{
    atom.func(atom.value, outputStream);
}
