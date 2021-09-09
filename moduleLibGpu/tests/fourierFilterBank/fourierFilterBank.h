#pragma once

#include "gpuModuleHeader.h"

namespace fourierFilterBank {

//================================================================
//
// Process
//
//================================================================

struct Process {};

//================================================================
//
// FourierFilterBank
//
//================================================================

class FourierFilterBank
{

public:

    FourierFilterBank();
    ~FourierFilterBank();

public:

    void serialize(const ModuleSerializeKit& kit);
    bool reallocValid() const;
    stdbool realloc(stdPars(GpuModuleReallocKit));
    bool active() const;
    stdbool process(const Process& o, stdPars(GpuModuleProcessKit));

private:

    DynamicClass<class FourierFilterBankImpl> instance;

};

//----------------------------------------------------------------

}
