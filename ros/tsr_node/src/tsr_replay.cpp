// Replay node (test harness): read PNG sequence -> BGR -> NV12 -> publish
// Nv12Image at 10 Hz (0.1s inter-frame). Stand-in for the real camera driver.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "tsr_node/msg/nv12_image.hpp"
#include "tsr/nv12.hpp"

using namespace tsr;

static int frame_index(const std::string& name) {
    size_t dot = name.rfind('.');
    size_t us = name.rfind('_', dot == std::string::npos ? name.size() : dot);
    if (us == std::string::npos) return -1;
    return std::atoi(name.substr(us + 1, (dot == std::string::npos ? name.size() : dot) - us - 1).c_str());
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("tsr_replay");
    auto pub = node->create_publisher<tsr_node::msg::Nv12Image>("image_nv12", 10);

    std::string dir = "/home/sunrise/tsr_test/self/images/test/20260907_5";
    double fps = 10.0;
    int loop = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--fps" && i + 1 < argc) fps = std::atof(argv[++i]);
        else if (a == "--loop" && i + 1 < argc) loop = std::atoi(argv[++i]);
    }

    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (d) {
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            std::string n = e->d_name;
            if (n.size() > 4 && (n.substr(n.size() - 4) == ".png" || n.substr(n.size() - 4) == ".jpg"))
                files.push_back(dir + "/" + n);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
        int ia = frame_index(a), ib = frame_index(b);
        return ia == ib ? a < b : ia < ib;
    });
    RCLCPP_INFO(node->get_logger(), "replay %zu images @ %.1f Hz", files.size(), fps);

    auto frame_dur = std::chrono::microseconds((int64_t)(1e6 / fps));
    for (int lp = 0; lp < loop; lp++) {
        for (const auto& f : files) {
            auto t0 = std::chrono::steady_clock::now();
            cv::Mat bgr = cv::imread(f);
            if (bgr.empty()) continue;
            cv::resize(bgr, bgr, cv::Size(640, 352));
            auto nv12 = bgr2nv12(bgr);
            auto msg = std::make_unique<tsr_node::msg::Nv12Image>();
            msg->width = 640;
            msg->height = 352;
            msg->data.assign(nv12.begin(), nv12.end());
            msg->header.stamp = node->now();
            pub->publish(std::move(msg));

            auto t1 = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
            if (elapsed < frame_dur)
                std::this_thread::sleep_for(frame_dur - elapsed);
        }
    }
    rclcpp::shutdown();
    return 0;
}
