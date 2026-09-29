// Classifier: RDK X5 BPU tiny CNN (28 classes, incl. 'other'), 64x64 NV12 input.
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "dnn/hb_dnn.h"

namespace tsr {

class Classifier {
  public:
    Classifier(const std::string& model_path);
    ~Classifier();

    int num_classes() const { return num_classes_; }

    // nv12: packed NV12 64x64 (6144 bytes). Returns num_classes() logits.
    std::vector<float> run(const uint8_t* nv12);

  private:
    bool init(const std::string& path);
    void deinit();

    hbPackedDNNHandle_t packed_ = nullptr;
    hbDNNHandle_t dnn_ = nullptr;
    hbDNNTensorProperties input_prop_{}, output_prop_{};
    hbDNNTensor input_{}, output_{};
    int input_h_ = 0, input_w_ = 0;
    int num_classes_ = 0;
};

}  // namespace tsr
