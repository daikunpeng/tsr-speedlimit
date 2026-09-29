#include "tsr/tracker.hpp"
#include <algorithm>
#include <cmath>
#include "tsr/class_names.hpp"

namespace tsr {

Tracker::Tracker(int N, int M, int K, float assoc_gate, int other_id)
    : N_(N), M_(M), K_(K), assoc_gate_(assoc_gate), other_id_(other_id) {}

// Center-distance normalized by size: lower is better. More robust than IoU for
// small fast-moving signs (IoU collapses when the box shifts/grows between frames).
float Tracker::dist_score(const Track& t, const DetResult& d) const {
    float ctx = (t.x1 + t.x2) / 2, cty = (t.y1 + t.y2) / 2;
    float cdx = (d.x1 + d.x2) / 2, cdy = (d.y1 + d.y2) / 2;
    float dist = std::sqrt((ctx - cdx) * (ctx - cdx) + (cty - cdy) * (cty - cdy));
    float st = std::max(t.x2 - t.x1, t.y2 - t.y1);
    float sd = std::max(d.x2 - d.x1, d.y2 - d.y1);
    return dist / std::max(st, sd);
}

std::vector<TrackEvent> Tracker::update(const std::vector<DetResult>& dets, int frame_idx) {
    std::vector<TrackEvent> events;
    std::vector<int> active;
    for (auto& kv : tracks_)
        if (kv.second.state != 2) active.push_back(kv.first);  // not lost

    std::vector<bool> det_matched(dets.size(), false);
    std::vector<bool> track_matched(active.size(), false);

    // Greedy nearest association by center-distance gate.
    for (size_t di = 0; di < dets.size(); di++) {
        int best = -1;
        float best_score = assoc_gate_;
        for (size_t ti = 0; ti < active.size(); ti++) {
            if (track_matched[ti]) continue;
            float s = dist_score(tracks_[active[ti]], dets[di]);
            if (s < best_score) { best_score = s; best = ti; }
        }
        if (best >= 0) {
            track_matched[best] = true;
            det_matched[di] = true;
            Track& t = tracks_[active[best]];
            t.x1 = dets[di].x1; t.y1 = dets[di].y1;
            t.x2 = dets[di].x2; t.y2 = dets[di].y2;
            t.det_streak++;
            t.miss_streak = 0;

            // Vote (speed value): track consecutive same cls_id.
            if (dets[di].cls_id == t.vote_val) {
                t.vote_streak++;
            } else {
                t.vote_val = dets[di].cls_id;
                t.vote_streak = 1;
            }

            // State transition candidate -> confirmed.
            if (t.state == 0 && t.det_streak >= N_) {
                t.state = 1;
                events.push_back({frame_idx, t.id, "confirmed", ""});
            }

            // Emit speed when confirmed and K consecutive same non-other vote.
            if (t.state == 1 && t.vote_streak >= K_ && t.vote_val != other_id_) {
                std::string name = (t.vote_val >= 0 && t.vote_val < 28)
                                       ? kClsNames[t.vote_val] : "?";
                if (name != t.emitted) {
                    t.emitted = name;
                    events.push_back({frame_idx, t.id, "speed", name});
                }
            }
        }
    }

    // New candidate tracks for unmatched detections.
    for (size_t di = 0; di < dets.size(); di++) {
        if (det_matched[di]) continue;
        Track t;
        t.id = next_id_++;
        t.state = 0;
        t.det_streak = 1;
        t.x1 = dets[di].x1; t.y1 = dets[di].y1;
        t.x2 = dets[di].x2; t.y2 = dets[di].y2;
        t.vote_val = dets[di].cls_id;
        t.vote_streak = 1;
        tracks_[t.id] = t;
        events.push_back({frame_idx, t.id, "new", ""});
    }

    // Handle unmatched tracks (miss -> candidate reset / confirmed->lost / lost->deleted).
    std::vector<int> to_delete;
    for (size_t ti = 0; ti < active.size(); ti++) {
        if (track_matched[ti]) continue;
        Track& t = tracks_[active[ti]];
        t.miss_streak++;
        t.det_streak = 0;
        if (t.state == 0) {
            if (t.miss_streak >= M_) {  // candidate: M misses -> drop
                to_delete.push_back(t.id);
                events.push_back({frame_idx, t.id, "deleted", ""});
            }
        } else if (t.state == 1 && t.miss_streak >= M_) {
            t.state = 2;  // confirmed -> lost
            events.push_back({frame_idx, t.id, "lost", ""});
        } else if (t.state == 2 && t.miss_streak >= M_ * 4) {
            to_delete.push_back(t.id);
            events.push_back({frame_idx, t.id, "deleted", ""});
        }
    }
    for (int id : to_delete) tracks_.erase(id);

    return events;
}

std::vector<TrackState> Tracker::get_tracks() const {
    std::vector<TrackState> out;
    for (const auto& kv : tracks_) {
        const Track& t = kv.second;
        TrackState s;
        s.track_id = t.id;
        s.state = t.state;
        s.x1 = t.x1; s.y1 = t.y1; s.x2 = t.x2; s.y2 = t.y2;
        s.speed = t.emitted;
        out.push_back(s);
    }
    return out;
}

}  // namespace tsr
