#include "../utils.h"
#include "operators.h"
#include "ops/clip/clip.h"

#ifdef ENABLE_CPU
#include "cpu/clip_cpu.h"
#endif


__C infiniopStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input,
    infiniopTensorDescriptor_t min,
    infiniopTensorDescriptor_t max,
    infiniopTensorDescriptor_t output) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateClipDescriptor(handle, (ClipCpuDescriptor_t *) desc_ptr, input, min, max, output);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc, void *input, void const *min, void const *max,void* output ,void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuClip((ClipCpuDescriptor_t) desc, input, min, max, output, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyClipDescriptor((ClipCpuDescriptor_t) desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}
