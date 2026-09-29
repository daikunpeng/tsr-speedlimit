// TSR NV12 helpers (Level 1 uses BGR->NV12 only; NV12 crop/resize added for ROS).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>

namespace tsr {

// Packed NV12: Y plane (H*W) + interleaved UV (H*W/2), total H*W*3/2 bytes.
inline std::vector<uint8_t> bgr2nv12(const cv::Mat& bgr) {
    int h = bgr.rows, w = bgr.cols;
    cv::Mat yuv;
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420);
    std::vector<uint8_t> out(h * w * 3 / 2);
    uint8_t* yuv_ptr = yuv.ptr<uint8_t>();
    uint8_t* dst = out.data();
    int y_size = h * w;
    memcpy(dst, yuv_ptr, y_size);                      // Y
    int uv_h = h / 2, uv_w = w / 2;
    uint8_t* u = yuv_ptr + y_size;
    uint8_t* v = u + uv_h * uv_w;
    uint8_t* dst_uv = dst + y_size;
    for (int i = 0; i < uv_w * uv_h; i++) {
        *dst_uv++ = *u++;
        *dst_uv++ = *v++;
    }
    return out;
}

// Full-frame packed NV12 -> BGR (used only when detections exist, for the
// classifier crop path; mirrors the validated BGR crop pipeline).
inline cv::Mat nv12_to_bgr(const uint8_t* nv12, int W, int H) {
    cv::Mat y(H, W, CV_8UC1, (void*)nv12);
    cv::Mat uv(H / 2, W, CV_8UC1, (void*)(nv12 + W * H));
    cv::Mat u(H / 2, W / 2, CV_8UC1), v(H / 2, W / 2, CV_8UC1);
    for (int r = 0; r < H / 2; r++)
        for (int c = 0; c < W / 2; c++) {
            u.at<uint8_t>(r, c) = uv.at<uint8_t>(r, 2 * c);
            v.at<uint8_t>(r, c) = uv.at<uint8_t>(r, 2 * c + 1);
        }
    cv::resize(u, u, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    cv::resize(v, v, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    cv::Mat yuv;
    std::vector<cv::Mat> ch = {y, u, v};
    cv::merge(ch, yuv);
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR);
    return bgr;
}

// Crop a region of packed NV12 and convert ONLY that crop to BGR (avoids the
// full-frame conversion; used in the hot path). Coordinates are even-aligned.
inline cv::Mat nv12_crop_to_bgr(const uint8_t* nv12, int W, int H,
                                int x1, int y1, int x2, int y2) {
    x1 &= ~1; y1 &= ~1;
    x2 = std::min(W, x2) & ~1;
    y2 = std::min(H, y2) & ~1;
    int cw = x2 - x1, ch = y2 - y1;
    if (cw < 2 || ch < 2) return cv::Mat();

    // Build a small contiguous NV12: Y crop + interleaved UV crop.
    cv::Mat nv12_crop(ch * 3 / 2, cw, CV_8UC1);
    const uint8_t* y = nv12;
    const uint8_t* uv = nv12 + W * H;
    uint8_t* dst = nv12_crop.ptr<uint8_t>();
    for (int r = 0; r < ch; r++)
        memcpy(dst + r * cw, y + (y1 + r) * W + x1, cw);
    dst += cw * ch;
    for (int r = 0; r < ch / 2; r++)
        memcpy(dst + r * cw, uv + (y1 / 2 + r) * W + x1, cw);

    cv::Mat bgr;
    cv::cvtColor(nv12_crop, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}

}  // namespace tsr
