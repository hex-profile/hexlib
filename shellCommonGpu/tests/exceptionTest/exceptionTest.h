#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"

namespace exceptionTest {

//================================================================
//
// ExceptionTest
//
//================================================================

struct ExceptionTest
{
    static UniquePtr<ExceptionTest> create();
    virtual ~ExceptionTest() {}

    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual bool active() const =0;
    virtual void process(stdPars(GpuModuleProcessKit)) =0;
};

//----------------------------------------------------------------

}

using exceptionTest::ExceptionTest;
