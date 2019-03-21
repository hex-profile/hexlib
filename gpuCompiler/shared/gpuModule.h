#pragma once

//================================================================
//
// When GPU module is compiled, the binary data is translated to C file.
//
// A static module descriptor structure is filled,
// specifying the module's binary data and the list of kernel names.
//
// Pointers to all module descriptors are merged into
// the special data section.
//
// For each kernel, a static kernel descriptor structure is generated,
// containing pointer to the parent's module descriptor pointer
// and the kernel index inside the module.
//
// When the startup loads modules, it goes thru all module descriptors,
// loads module data (filling runtime internal structures)
// and loads kernel objects from modules, using kernel name list.
//
// When the user calls a kernel, specifying the kernel's static descriptor,
// the runtime uses the kernel descriptor's pointer to the module descriptor pointer,
// to compute the index of the modile in global array (merged section).
// Also the runtime reads the kernel's local index (in module),
// so it has both indices: module index and kernel index,
// for convenient implementation.
//
//================================================================

//================================================================
//
// GpuKernelLink
//
// CUDA version
//
//================================================================

#define GPU_DEFINE_KERNEL_LINK \
    \
    struct GpuKernelLink \
    { \
        const struct GpuModuleDesc* const* parentModule; \
        int indexInModule; \
    };

GPU_DEFINE_KERNEL_LINK

//================================================================
//
// GpuSamplerLink
//
// CUDA version
//
//================================================================

#define GPU_DEFINE_SAMPLER_LINK \
    \
    struct GpuSamplerLink \
    { \
        const struct GpuModuleDesc* const* parentModule; \
        int indexInModule; \
    };

GPU_DEFINE_SAMPLER_LINK

//================================================================
//
// GpuModuleDesc
//
//================================================================

#define GPU_DEFINE_MODULE_DESC \
    \
    struct GpuModuleDesc \
    { \
        const unsigned char* binDataPtr; \
        int binDataSize; \
        const char* const* kernelNames; \
        int kernelCount; \
        const char* const* samplerNames; \
        int samplerCount; \
    };

GPU_DEFINE_MODULE_DESC
