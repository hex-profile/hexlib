#include "gpuModuleKeeper.h"

#include "shared/gpuModule.h"
#include "dataAlloc/arrayObjMem.inl"
#include "storage/rememberCleanup.h"

//================================================================
//
// GPUBIN_FOREACH_BEG
// GPUBIN_FOREACH_END
//
//================================================================

#if defined(_MSC_VER)

    #pragma section("gpu_section$a", read)
    #pragma section("gpu_section$z", read)
    #pragma section("gpu_section$m", read)

    __declspec(allocate("gpu_section$a")) const GpuModuleDesc* const gpuSectionStart = 0;
    __declspec(allocate("gpu_section$z")) const GpuModuleDesc* const gpuSectionEnd = 0;

    #pragma comment(linker, "/merge:gpu_section=.rdata")

#elif defined(__GNUC__)

    const GpuModuleDesc* atLeastOneMember __attribute__((section("gpu_section"))) = nullptr;

    extern const GpuModuleDesc* const __start_gpu_section;
    extern const GpuModuleDesc* const __stop_gpu_section;

    #define gpuSectionStart __start_gpu_section
    #define gpuSectionEnd __stop_gpu_section

#else

    #error

#endif

//================================================================
//
// GpuModuleKeeper::create
//
//================================================================

bool GpuModuleKeeper::create(const GpuContext& context, stdPars(CreateKit))
{
    stdBegin;

    destroy();

    //----------------------------------------------------------------
    //
    // Count modules, count kernels, count samplers.
    //
    // Build modref index -> kernel index map.
    // Build modref index -> sampler index map.
    //
    //----------------------------------------------------------------

    const GpuModuleDesc* const* modrefPtr = &gpuSectionStart;
    ptrdiff_t modrefCountEx = &gpuSectionEnd - &gpuSectionStart;

    Space modrefCount = 0;
    REQUIRE(convertExact(modrefCountEx, modrefCount));
    REQUIRE(modrefCount >= 0);

    ////

    require(modrefToKernelIndex.realloc(modrefCount, cpuBaseByteAlignment, kit.malloc, stdPass));
    ARRAY_EXPOSE(modrefToKernelIndex);
    REMEMBER_CLEANUP1_EX(kernelIndexCleanup, modrefToKernelIndex.dealloc(), ArrayMemory<Space>&, modrefToKernelIndex);

    require(modrefToSamplerIndex.realloc(modrefCount, cpuBaseByteAlignment, kit.malloc, stdPass));
    ARRAY_EXPOSE(modrefToSamplerIndex);
    REMEMBER_CLEANUP1_EX(samplerIndexCleanup, modrefToSamplerIndex.dealloc(), ArrayMemory<Space>&, modrefToSamplerIndex);

    ////

    int32 moduleCount = 0;
    int32 kernelCount = 0;
    int32 samplerCount = 0;

    ////

    for (Space k = 0; k < modrefCount; ++k)
    {
        modrefToKernelIndexPtr[k] = kernelCount;
        modrefToSamplerIndexPtr[k] = samplerCount;

        if (modrefPtr[k] != 0) // Because Microsoft expands the section and fills with zeroes
        {
            kernelCount += modrefPtr[k]->kernelCount;
            samplerCount += modrefPtr[k]->samplerCount;
            ++moduleCount;
        }
    }

    //----------------------------------------------------------------
    //
    // Allocate memory
    //
    //----------------------------------------------------------------

    require(moduleInfo.realloc(moduleCount, kit.malloc, true, stdPass));
    ARRAY_EXPOSE(moduleInfo);
    REMEMBER_CLEANUP1_EX(moduleInfoCleanup, moduleInfo.dealloc(), ArrayObjMem<ModuleInfo>&, moduleInfo);

    ////

    require(kernelHandle.realloc(kernelCount, kit.malloc, true, stdPass));
    ARRAY_EXPOSE(kernelHandle);
    REMEMBER_CLEANUP1_EX(kernelHandleCleanup, kernelHandle.dealloc(), ArrayObjMem<GpuKernel>&, kernelHandle);

    require(kernelInfo.realloc(kernelCount, kit.malloc, true, stdPass));
    ARRAY_EXPOSE(kernelInfo);
    REMEMBER_CLEANUP1_EX(kernelInfoCleanup, kernelInfo.dealloc(), ArrayObjMem<KernelInfo>&, kernelInfo);

    ////

    require(samplerHandle.realloc(samplerCount, kit.malloc, true, stdPass));
    ARRAY_EXPOSE(samplerHandle);
    REMEMBER_CLEANUP1_EX(samplerHandleCleanup, samplerHandle.dealloc(), ArrayObjMem<GpuSampler>&, samplerHandle);

    require(samplerInfo.realloc(samplerCount, kit.malloc, true, stdPass));
    ARRAY_EXPOSE(samplerInfo);
    REMEMBER_CLEANUP1_EX(samplerInfoCleanup, samplerInfo.dealloc(), ArrayObjMem<SamplerInfo>&, samplerInfo);

    //----------------------------------------------------------------
    //
    // Load modules, fetch kernels, fetch samplers.
    //
    //----------------------------------------------------------------

    Space moduleIdx = 0;
    Space kernelIdx = 0;
    Space samplerIdx = 0;

    ////

    for (Space k = 0; k < modrefCount; ++k)
    {

        if (modrefPtr[k] != 0) // (Microsoft expands the section and fills with zeroes, on GCC we also have one NULL element)
        {

            //
            // Load module
            //

            const GpuModuleDesc& modDesc = *modrefPtr[k];

            ////

            Array<const uint8> binary(modDesc.binDataPtr, modDesc.binDataSize);

            ////

            REQUIRE(moduleIdx < moduleCount);
            ModuleInfo& modInfo = moduleInfoPtr[moduleIdx++];

            ////

            require(kit.gpuModuleCreation.createModuleFromBinary(context, binary, modInfo, stdPass));

            //
            // Load kernels
            //

            for (Space k = 0; k < modDesc.kernelCount; ++k)
            {
                REQUIRE(kernelIdx < kernelCount);

                ////

                require(kit.gpuKernelLoading.createKernelFromModule(modInfo, modDesc.kernelNames[k],
                    helpModify(kernelInfoPtr[kernelIdx]).owner, stdPass));

                helpModify(kernelInfoPtr[kernelIdx]).name = modDesc.kernelNames[k];

                ////

                kernelHandlePtr[kernelIdx] = helpModify(kernelInfoPtr[kernelIdx]).owner;

                ////

                ++kernelIdx;
            }

            //
            // Load samplers
            //

            for (Space k = 0; k < modDesc.samplerCount; ++k)
            {
                REQUIRE(samplerIdx < samplerCount);

                ////

                require(kit.gpuSamplerLoading.getSamplerFromModule(modInfo, modDesc.samplerNames[k],
                    helpModify(samplerInfoPtr[samplerIdx]).owner, stdPass));

                helpModify(samplerInfoPtr[samplerIdx]).name = modDesc.samplerNames[k];

                samplerHandlePtr[samplerIdx] = helpModify(samplerInfoPtr[samplerIdx]).owner;

                ////

                ++samplerIdx;
            }
        }
    }

    ////

    REQUIRE(moduleIdx == moduleCount);
    REQUIRE(kernelIdx == kernelCount);
    REQUIRE(samplerIdx == samplerCount);

    //
    // Record success
    //

    kernelIndexCleanup.cancel();
    samplerIndexCleanup.cancel();
    moduleInfoCleanup.cancel();
    kernelHandleCleanup.cancel();
    kernelInfoCleanup.cancel();
    samplerHandleCleanup.cancel();
    samplerInfoCleanup.cancel();

    loaded = true;

    stdEnd;
}

//================================================================
//
// GpuModuleKeeper::destroy
//
//================================================================

void GpuModuleKeeper::destroy()
{
    samplerInfo.dealloc();
    samplerHandle.dealloc();

    kernelInfo.dealloc();
    kernelHandle.dealloc();

    moduleInfo.dealloc();

    modrefToSamplerIndex.dealloc();
    modrefToKernelIndex.dealloc();

    loaded = false;
}

//================================================================
//
// GpuModuleKeeper::fetchKernel
//
//================================================================

bool GpuModuleKeeper::fetchKernel(const GpuKernelLink& link, GpuKernel& kernel, stdPars(ErrorLogKit)) const
{
    stdBegin;

    REQUIRE(loaded);

    const GpuKernelLink& desc = link;

    ////

    const GpuModuleDesc* const* moduleDesc = desc.parentModule;
    Space modrefIdx = Space(moduleDesc - &gpuSectionStart);

    ARRAY_EXPOSE(modrefToKernelIndex);
    REQUIRE(SpaceU(modrefIdx) < SpaceU(modrefToKernelIndexSize));

    Space kernelIdx = modrefToKernelIndexPtr[modrefIdx] + desc.indexInModule;

    ////

    ARRAY_EXPOSE(kernelHandle);
    REQUIRE(SpaceU(kernelIdx) < SpaceU(kernelHandleSize));
    kernel = kernelHandlePtr[kernelIdx];

    stdEnd;
}

//================================================================
//
// GpuModuleKeeper::fetchSampler
//
//================================================================

bool GpuModuleKeeper::fetchSampler(const GpuSamplerLink& link, GpuSampler& sampler, stdPars(ErrorLogKit)) const
{
    stdBegin;

    REQUIRE(loaded);

    const GpuSamplerLink& desc = link;

    ////

    const GpuModuleDesc* const* moduleDesc = desc.parentModule;
    Space modrefIdx = Space(moduleDesc - &gpuSectionStart);

    ARRAY_EXPOSE(modrefToSamplerIndex);
    REQUIRE(SpaceU(modrefIdx) < SpaceU(modrefToSamplerIndexSize));

    Space samplerIdx = modrefToSamplerIndexPtr[modrefIdx] + desc.indexInModule;

    ////

    ARRAY_EXPOSE(samplerHandle);
    REQUIRE(SpaceU(samplerIdx) < SpaceU(samplerHandleSize));
    sampler = samplerHandlePtr[samplerIdx];

    stdEnd;
}
