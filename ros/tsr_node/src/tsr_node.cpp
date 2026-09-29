// TSR ROS node: subscribe NV12 image -> detector -> crop -> classifier -> tracker
// -> publish SpeedLimitSignArray. Business flow kept explicit for integration.
// Tuning params (score_thres/N/M/K/assoc_gate) loaded from a ROS2 YAML params file.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "tsr_node/msg/nv12_image.hpp"
#include "tsr_node/msg/speed_limit_sign.hpp"
#include "tsr_node/msg/speed_limit_sign_array.hpp"
#include "tsr/detector.hpp"
#include "tsr/classifier.hpp"
#include "tsr/tracker.hpp"
#include "tsr/nv12.hpp"

using namespace tsr;

class TsrNode : public rclcpp::Node {
  public:
    TsrNode(const std::string& det_path, const std::string& cls_path)
        : Node("tsr_node") {
        this->declare_parameter<double>("score_thres", 0.25);
        this->declare_parameter<int>("N", 3);
        this->declare_parameter<int>("M", 3);
        this->declare_parameter<int>("K", 3);
        this->declare_parameter<double>("assoc_gate", 2.0);

        float score_thres = this->get_parameter("score_thres").as_double();
        int N = this->get_parameter("N").as_int();
        int M = this->get_parameter("M").as_int();
        int K = this->get_parameter("K").as_int();
        float assoc_gate = this->get_parameter("assoc_gate").as_double();

        det_ = std::make_unique<Detector>(det_path, score_thres);
        cls_ = std::make_unique<Classifier>(cls_path);
        tracker_ = std::make_unique<Tracker>(N, M, K, assoc_gate);

        sub_ = this->create_subscription<tsr_node::msg::Nv12Image>(
            "image_nv12", 10,
            std::bind(&TsrNode::on_image, this, std::placeholders::_1));
        pub_ = this->create_publisher<tsr_node::msg::SpeedLimitSignArray>(
            "speed_limit_signs", 10);
        RCLCPP_INFO(this->get_logger(),
                    "TSR node ready (det %dx%d, cls %d classes, score_thres=%.2f N/M/K=%d/%d/%d)",
                    det_->input_w(), det_->input_h(), cls_->num_classes(),
                    score_thres, N, M, K);
    }

  private:
    void on_image(const tsr_node::msg::Nv12Image::SharedPtr msg) {
        const int W = det_->input_w(), H = det_->input_h();
        if ((int)msg->width != W || (int)msg->height != H) {
            RCLCPP_WARN(this->get_logger(), "expected %dx%d NV12, got %dx%d", W, H,
                        msg->width, msg->height);
            return;
        }
        if (msg->data.size() < (size_t)W * H * 3 / 2) return;

        // 1. Detector.
        auto boxes = det_->run(msg->data.data());

        // 2. Crop-first + classify (per-box NV12 crop -> BGR -> resize -> NV12).
        std::vector<DetResult> dets;
        for (auto& b : boxes) {
            int x1 = std::max(0, (int)std::floor(b.x1)), y1 = std::max(0, (int)std::floor(b.y1));
            int x2 = std::min(W, (int)std::ceil(b.x2)), y2 = std::min(H, (int)std::ceil(b.y2));
            if (x2 <= x1 || y2 <= y1) continue;
            cv::Mat crop_bgr = nv12_crop_to_bgr(msg->data.data(), W, H, x1, y1, x2, y2);
            if (crop_bgr.empty()) continue;
            cv::Mat crop64;
            cv::resize(crop_bgr, crop64, cv::Size(64, 64));
            auto cnv12 = bgr2nv12(crop64);
            auto logits = cls_->run(cnv12.data());
            int cid = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

            DetResult d;
            d.x1 = b.x1; d.y1 = b.y1; d.x2 = b.x2; d.y2 = b.y2;
            d.det_conf = b.score; d.cls_id = cid;
            dets.push_back(d);
        }

        // 3. Track (frame index from header stamp).
        int frame_idx = msg->header.stamp.sec * 1000 + msg->header.stamp.nanosec / 1000000;
        tracker_->update(dets, frame_idx);

        // 4. Publish.
        tsr_node::msg::SpeedLimitSignArray out;
        out.header = msg->header;
        for (auto& t : tracker_->get_tracks()) {
            tsr_node::msg::SpeedLimitSign s;
            s.track_id = t.track_id;
            s.x1 = t.x1; s.y1 = t.y1; s.x2 = t.x2; s.y2 = t.y2;
            s.state = t.state;
            s.speed_value = t.speed;
            out.signs.push_back(s);
        }
        pub_->publish(out);
    }

    std::unique_ptr<tsr::Detector> det_;
    std::unique_ptr<tsr::Classifier> cls_;
    std::unique_ptr<tsr::Tracker> tracker_;
    rclcpp::Subscription<tsr_node::msg::Nv12Image>::SharedPtr sub_;
    rclcpp::Publisher<tsr_node::msg::SpeedLimitSignArray>::SharedPtr pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    std::string det = "/home/sunrise/tsr_models/det_cropcolor1_352x640_bayese_640x352_nv12.bin";
    std::string cls = "/home/sunrise/tsr_models/cls_v3_64x64_nv12_64x64.bin";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--det" && i + 1 < argc) det = argv[++i];
        else if (a == "--cls" && i + 1 < argc) cls = argv[++i];
    }
    auto node = std::make_shared<TsrNode>(det, cls);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
