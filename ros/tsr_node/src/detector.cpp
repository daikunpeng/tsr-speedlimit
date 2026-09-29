#include "tsr/detector.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace tsr {

static int32_t check_ret(int32_t rc, const char* what) {
    if (rc != 0) std::cerr << "[Detector] " << what << " rc=" << rc << std::endl;
    return rc;
}

Detector::Detector(const std::string& model_path, float score_thres, float nms_thres, int reg)
    : score_thres_(score_thres), nms_thres_(nms_thres), reg_(reg) {
    if (!init(model_path)) {
        std::cerr << "[Detector] init failed" << std::endl;
        throw std::runtime_error("Detector init failed");
    }
}

bool Detector::init(const std::string& path) {
    const char* file = path.c_str();
    if (check_ret(hbDNNInitializeFromFiles(&packed_, &file, 1), "init from files")) return false;

    const char** names = nullptr;
    int count = 0;
    if (check_ret(hbDNNGetModelNameList(&names, &count, packed_), "get model names")) return false;
    if (check_ret(hbDNNGetModelHandle(&dnn_, packed_, names[0]), "get model handle")) return false;

    int in_count = 0;
    hbDNNGetInputCount(&in_count, dnn_);
    hbDNNGetInputTensorProperties(&input_prop_, dnn_, 0);
    if (input_prop_.tensorType != HB_DNN_IMG_TYPE_NV12) {
        std::cerr << "[Detector] input is not NV12" << std::endl;
        return false;
    }
    input_h_ = input_prop_.validShape.dimensionSize[2];
    input_w_ = input_prop_.validShape.dimensionSize[3];

    int out_count = 0;
    hbDNNGetOutputCount(&out_count, dnn_);
    if (out_count != 6) {
        std::cerr << "[Detector] expected 6 outputs, got " << out_count << std::endl;
        return false;
    }

    // Reorder 6 outputs to [cls0,bbox0,cls1,bbox1,cls2,bbox2] by shape.
    const int H[3] = {input_h_ / 8, input_h_ / 16, input_h_ / 32};
    const int W[3] = {input_w_ / 8, input_w_ / 16, input_w_ / 32};
    order_.assign(6, -1);
    for (int s = 0; s < 3; s++) {
        int cls_expected = 1, bbox_expected = reg_ * 4;
        for (int j = 0; j < out_count; j++) {
            hbDNNTensorProperties p;
            hbDNNGetOutputTensorProperties(&p, dnn_, j);
            int h = p.validShape.dimensionSize[1];
            int w = p.validShape.dimensionSize[2];
            int c = p.validShape.dimensionSize[3];
            if (h == H[s] && w == W[s]) {
                if (c == cls_expected && order_[s * 2] < 0) order_[s * 2] = j;
                else if (c == bbox_expected && order_[s * 2 + 1] < 0) order_[s * 2 + 1] = j;
            }
        }
    }

    // Allocate input/output buffers once.
    int in_size = input_h_ * input_w_ * 3 / 2;
    input_.properties = input_prop_;
    hbSysAllocCachedMem(&input_.sysMem[0], in_size);

    outputs_.resize(out_count);
    for (int i = 0; i < out_count; i++) {
        hbDNNGetOutputTensorProperties(&outputs_[i].properties, dnn_, i);
        hbSysAllocCachedMem(&outputs_[i].sysMem[0], outputs_[i].properties.alignedByteSize);
    }
    std::cout << "[Detector] loaded " << path << " input " << input_w_ << "x" << input_h_
              << " order [" << order_[0] << "," << order_[1] << "," << order_[2] << ","
              << order_[3] << "," << order_[4] << "," << order_[5] << "]" << std::endl;
    return true;
}

void Detector::deinit() {
    for (auto& o : outputs_) hbSysFreeMem(&o.sysMem[0]);
    hbSysFreeMem(&input_.sysMem[0]);
    if (packed_) hbDNNRelease(packed_);
}

Detector::~Detector() { deinit(); }

static float iou(const DetBox& a, const DetBox& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float aa = (a.x2 - a.x1) * (a.y2 - a.y1);
    float bb = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (aa + bb - inter + 1e-9f);
}

std::vector<DetBox> Detector::run(const uint8_t* nv12) {
    int in_size = input_h_ * input_w_ * 3 / 2;
    memcpy(input_.sysMem[0].virAddr, nv12, in_size);
    hbSysFlushMem(&input_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    hbDNNTaskHandle_t task = nullptr;
    hbDNNInferCtrlParam ctrl;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
    hbDNNTensor* out_ptr = outputs_.data();
    hbDNNInfer(&task, &out_ptr, &input_, dnn_, &ctrl);
    hbDNNWaitTaskDone(task, 0);
    hbDNNReleaseTask(task);

    const int H[3] = {input_h_ / 8, input_h_ / 16, input_h_ / 32};
    const int W[3] = {input_w_ / 8, input_w_ / 16, input_w_ / 32};
    const int strides[3] = {8, 16, 32};
    float conf_raw = -std::log(1.0f / score_thres_ - 1.0f);

    std::vector<DetBox> cands;
    for (int s = 0; s < 3; s++) {
        int cls_idx = order_[s * 2], bbox_idx = order_[s * 2 + 1];
        hbSysFlushMem(&outputs_[cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&outputs_[bbox_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        const float* cls = reinterpret_cast<const float*>(outputs_[cls_idx].sysMem[0].virAddr);
        const float* bbox = reinterpret_cast<const float*>(outputs_[bbox_idx].sysMem[0].virAddr);
        int h = H[s], w = W[s];
        // layout (1,h,w,C) row-major: index = (gh*w + gw)*C
        for (int gh = 0; gh < h; gh++) {
            for (int gw = 0; gw < w; gw++) {
                int cell = gh * w + gw;
                float logit = cls[cell * 1];  // single class
                if (logit < conf_raw) continue;
                float score = 1.0f / (1.0f + std::exp(-logit));
                // DFL decode
                const float* d = bbox + cell * (reg_ * 4);
                float ltrb[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < reg_; j++) {
                        float e = std::exp(d[i * reg_ + j]);
                        ltrb[i] += e * j;
                        sum += e;
                    }
                    ltrb[i] /= sum;
                }
                DetBox b;
                b.x1 = (gw + 0.5f - ltrb[0]) * strides[s];
                b.y1 = (gh + 0.5f - ltrb[1]) * strides[s];
                b.x2 = (gw + 0.5f + ltrb[2]) * strides[s];
                b.y2 = (gh + 0.5f + ltrb[3]) * strides[s];
                b.score = score;
                cands.push_back(b);
            }
        }
    }

    // Greedy NMS (single class), sorted by score desc.
    std::sort(cands.begin(), cands.end(),
              [](const DetBox& a, const DetBox& b) { return a.score > b.score; });
    std::vector<DetBox> kept;
    for (auto& c : cands) {
        bool suppressed = false;
        for (auto& k : kept) {
            if (iou(c, k) >= nms_thres_) { suppressed = true; break; }
        }
        if (!suppressed) kept.push_back(c);
    }
    return kept;
}

}  // namespace tsr
