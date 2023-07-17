#pragma once

#include "gpuModuleHeader.h"
#include "storage/dynamicClass.h"
#include "kits/gpuRgbFrameKit.h"

//================================================================
//
// Rotation3DTest
//
//================================================================

class Rotation3DTest
{

public:

    Rotation3DTest();
    ~Rotation3DTest();

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const;

public:

    stdbool process(stdPars(GpuModuleProcessKit));

private:

    DynamicClass<class Rotation3DTestImpl> instance;

};
