// Level 1 no-ROS test: read images -> det -> crop -> cls -> (optional) tracker.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tsr/detector.hpp"
#include "tsr/classifier.hpp"
#include "tsr/tracker.hpp"
#include "tsr/nv12.hpp"
#include "tsr/class_names.hpp"

using namespace tsr;

static int frame_index(const std::string& name) {
    size_t dot = name.rfind('.');
    size_t us = name.rfind('_', dot == std::string::npos ? name.size() : dot);
    if (us == std::string::npos) return -1;
    return std::atoi(name.substr(us + 1, (dot == std::string::npos ? name.size() : dot) - us - 1).c_str());
}

static std::vector<std::string> list_images(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string n = e->d_name;
        if (n.size() < 4) continue;
        std::string ext = n.substr(n.size() - 4);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
            out.push_back(dir + "/" + n);
    }
    closedir(d);
    return out;
}

int main(int argc, char** argv) {
    std::string det_path, cls_path, dir, out_path, vis_dir;
    float score_thres = 0.25f;
    bool track_mode = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--det" && i + 1 < argc) det_path = argv[++i];
        else if (a == "--cls" && i + 1 < argc) cls_path = argv[++i];
        else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--score-thres" && i + 1 < argc) score_thres = std::atof(argv[++i]);
        else if (a == "--vis" && i + 1 < argc) vis_dir = argv[++i];
        else if (a == "--track") track_mode = true;
    }
    if (det_path.empty() || cls_path.empty() || dir.empty()) {
        std::cerr << "usage: tsr_test --det <bin> --cls <bin> --dir <dir> [--out f] "
                     "[--score-thres 0.25] [--vis <out_dir>] [--track]" << std::endl;
        return 1;
    }

    if (!vis_dir.empty()) {
        std::string mk = "mkdir -p " + vis_dir;
        std::system(mk.c_str());
    }

    Detector det(det_path, score_thres);
    Classifier cls(cls_path);
    Tracker tracker;

    // Gather + sort images by trailing frame index.
    std::vector<std::string> files = list_images(dir);
    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
        int ia = frame_index(a), ib = frame_index(b);
        return ia == ib ? a < b : ia < ib;
    });

    std::ofstream out;
    if (!out_path.empty()) out.open(out_path);

    int n_det = 0;
    for (const auto& f : files) {
        cv::Mat bgr = cv::imread(f);
        if (bgr.empty()) continue;
        // Direct resize to detector input (no-op for native 640x352).
        cv::Mat img = bgr;
        if (img.cols != det.input_w() || img.rows != det.input_h())
            cv::resize(img, img, cv::Size(det.input_w(), det.input_h()));

        auto nv12 = bgr2nv12(img);
        auto boxes = det.run(nv12.data());

        std::vector<DetResult> dets;
        for (auto& b : boxes) {
            int x1 = std::max(0, (int)std::floor(b.x1)), y1 = std::max(0, (int)std::floor(b.y1));
            int x2 = std::min(det.input_w(), (int)std::ceil(b.x2)), y2 = std::min(det.input_h(), (int)std::ceil(b.y2));
            if (x2 <= x1 || y2 <= y1) continue;
            cv::Mat crop = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            cv::Mat crop64;
            cv::resize(crop, crop64, cv::Size(64, 64));
            auto cnv12 = bgr2nv12(crop64);
            auto logits = cls.run(cnv12.data());
            int cid = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

            DetResult d;
            d.x1 = b.x1; d.y1 = b.y1; d.x2 = b.x2; d.y2 = b.y2;
            d.det_conf = b.score; d.cls_id = cid;
            dets.push_back(d);
            n_det++;

            if (!track_mode && out.is_open()) {
                std::string name = f.substr(f.rfind('/') + 1);
                out << name << " " << b.x1 << " " << b.y1 << " " << b.x2 << " " << b.y2
                    << " " << b.score << " " << cid << "\n";
            }
        }

        if (!vis_dir.empty() && !dets.empty()) {
            cv::Mat canvas = img.clone();
            for (auto& d : dets) {
                cv::rectangle(canvas, cv::Point((int)d.x1, (int)d.y1),
                              cv::Point((int)d.x2, (int)d.y2), cv::Scalar(0, 255, 0), 2);
                std::string name = (d.cls_id >= 0 && d.cls_id < 28) ? kClsNames[d.cls_id] : "?";
                char lbl[64];
                snprintf(lbl, sizeof(lbl), "%s %.2f", name.c_str(), d.det_conf);
                cv::putText(canvas, lbl, cv::Point((int)d.x1, std::max(0, (int)d.y1 - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            std::string out_img = vis_dir + "/" + f.substr(f.rfind('/') + 1);
            cv::imwrite(out_img, canvas);
        }

        if (track_mode) {
            int fi = frame_index(f);
            auto events = tracker.update(dets, fi);
            for (auto& e : events) {
                if (out.is_open())
                    out << e.frame_idx << " " << e.type << " " << e.track_id << " " << e.speed << "\n";
                else
                    std::cout << e.frame_idx << " " << e.type << " " << e.track_id << " " << e.speed << "\n";
            }
        }
    }

    std::cout << "processed " << files.size() << " images, " << n_det << " detections" << std::endl;
    return 0;
}
