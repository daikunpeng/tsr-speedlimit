#include "tsr/classifier.hpp"
#include <cstring>
#include <iostream>

namespace tsr {

Classifier::Classifier(const std::string& model_path) {
    if (!init(model_path)) throw std::runtime_error("Classifier init failed");
}

bool Classifier::init(const std::string& path) {
    const char* file = path.c_str();
    if (hbDNNInitializeFromFiles(&packed_, &file, 1)) return false;
    const char** names = nullptr;
    int count = 0;
    hbDNNGetModelNameList(&names, &count, packed_);
    if (hbDNNGetModelHandle(&dnn_, packed_, names[0])) return false;

    hbDNNGetInputTensorProperties(&input_prop_, dnn_, 0);
    input_h_ = input_prop_.validShape.dimensionSize[2];
    input_w_ = input_prop_.validShape.dimensionSize[3];

    hbDNNGetOutputTensorProperties(&output_prop_, dnn_, 0);
    num_classes_ = output_prop_.validShape.dimensionSize[1];

    int in_size = input_h_ * input_w_ * 3 / 2;
    input_.properties = input_prop_;
    hbSysAllocCachedMem(&input_.sysMem[0], in_size);

    output_.properties = output_prop_;
    hbSysAllocCachedMem(&output_.sysMem[0], output_prop_.alignedByteSize);

    std::cout << "[Classifier] loaded " << path << " input " << input_w_ << "x" << input_h_
              << " classes " << num_classes_ << std::endl;
    return true;
}

void Classifier::deinit() {
    hbSysFreeMem(&output_.sysMem[0]);
    hbSysFreeMem(&input_.sysMem[0]);
    if (packed_) hbDNNRelease(packed_);
}

Classifier::~Classifier() { deinit(); }

std::vector<float> Classifier::run(const uint8_t* nv12) {
    int in_size = input_h_ * input_w_ * 3 / 2;
    memcpy(input_.sysMem[0].virAddr, nv12, in_size);
    hbSysFlushMem(&input_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    hbDNNTaskHandle_t task = nullptr;
    hbDNNInferCtrlParam ctrl;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
    hbDNNTensor* out_ptr = &output_;
    hbDNNInfer(&task, &out_ptr, &input_, dnn_, &ctrl);
    hbDNNWaitTaskDone(task, 0);
    hbDNNReleaseTask(task);

    hbSysFlushMem(&output_.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    const float* logits = reinterpret_cast<const float*>(output_.sysMem[0].virAddr);
    return std::vector<float>(logits, logits + num_classes_);
}

}  // namespace tsr
