#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t input_data_size;
};

typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *,
                                         infiniopTensorDescriptor_t input,
                                         infiniopTensorDescriptor_t min,
                                         infiniopTensorDescriptor_t max,
                                         infiniopTensorDescriptor_t output);

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *input, void const *min, void const *max, void *output,
                         void *stream);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif
