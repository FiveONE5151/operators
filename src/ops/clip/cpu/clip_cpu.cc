#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include "status.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <ostream>

// TODO: implement the cpu algorithm
infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t input,
                                         infiniopTensorDescriptor_t min,
                                         infiniopTensorDescriptor_t max,
                                         infiniopTensorDescriptor_t output) {

    // all tensor should be f16 or f32
    if (input->dt != F16 && input->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (min->dt != F16 && min->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (max->dt != F16 && max->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // all tensor should be of same type
    if (!(input->dt == min->dt && input->dt == max->dt && min->dt == max->dt)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // min and max should be scalar
    if (min->ndim != 0 || max->ndim != 0) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    // std::cout << "input shape[0]: " << input->shape[0] << std::endl;
    uint64_t input_data_size = std::accumulate(input->shape, input->shape + input->ndim, 1, std::multiplies<uint64_t>());

    // std::cout << "input_data_size: " << input_data_size << std::endl;

    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        input->dt,
        input->ndim,
        input_data_size,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc,
                          void *input, void const *min, void const *max, void *output,
                          void *stream) {

    auto input_ = reinterpret_cast<Tdata *>(input);
    auto min_ = reinterpret_cast<const Tdata *>(min);
    auto max_ = reinterpret_cast<const Tdata *>(max);
    auto output_ = reinterpret_cast<Tdata *>(output);

    for (uint64_t i = 0; i < desc->input_data_size; ++i) {
        if constexpr (std::is_same<Tdata, uint16_t>::value) {

            auto input_f16 = f16_to_f32(input_[i]);
            auto min_f16 = f16_to_f32(*min_);
            auto max_f16 = f16_to_f32(*max_);

            // std::cout << "input_f16: " << input_f16 << std::endl;
            // std::cout << "min_f16: " << min_f16 << std::endl;
            // std::cout << "max_f16: " << max_f16 << std::endl;

            output_[i] = f32_to_f16(std::min(std::max(input_f16, min_f16), max_f16));
        } else {
            output_[i] = std::min(std::max(input_[i], *min_), *max_);
        }
    }
    return STATUS_SUCCESS;
}
// fp16, fp16, fp16
// fp16. fp16, fp32
// fp16, fp32, fp32
// fp32, fp32, fp32
//
infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *input, void const *min, void const *max, void *output,
                         void *stream) {

    if (desc->dtype == F16) {
        return clip_cpu<uint16_t>(desc, input, min, max, output, stream);
    }
    if (desc->dtype == F32) {
        return clip_cpu<float>(desc, input, min, max, output, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}