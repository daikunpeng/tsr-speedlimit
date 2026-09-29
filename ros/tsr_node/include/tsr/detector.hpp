// Detector: RDK X5 BPU YOLOv5u (anchor-free) single-class detector.
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "dnn/hb_dnn.h"

namespace tsr {

struct DetBox {
    float x1, y1, x2, y2, score;
};

class Detector {
  public:
    Detector(const std::string& model_path, float score_thres = 0.25f,
             float nms_thres = 0.70f, int reg = 16);
    ~Detector();

    int input_h() const { return input_h_; }
    int input_w() const { return input_w_; }

    // nv12: packed NV12, size input_h()*input_w()*3/2 bytes.
    std::vector<DetBox> run(const uint8_t* nv12);

  private:
    bool init(const std::string& path);
    void deinit();

    hbPackedDNNHandle_t packed_ = nullptr;
    hbDNNHandle_t dnn_ = nullptr;
    hbDNNTensorProperties input_prop_{};
    hbDNNTensor input_{};
    std::vector<hbDNNTensor> outputs_;
    std::vector<int> order_;   // 6 entries: reorder to [cls0,bbox0,cls1,bbox1,cls2,bbox2]
    int input_h_ = 0, input_w_ = 0;
    float score_thres_, nms_thres_;
    int reg_;
};

}  // namespace tsr
