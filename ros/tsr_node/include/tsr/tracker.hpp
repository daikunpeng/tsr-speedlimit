// Sign lifecycle tracker + multi-frame speed voting (self-written, IoU association).
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace tsr {

struct DetResult {
    float x1, y1, x2, y2, det_conf;
    int cls_id;   // classifier argmax, 27 == "other"
};

struct TrackEvent {
    int frame_idx;
    int track_id;
    std::string type;   // "new", "confirmed", "speed", "lost", "deleted"
    std::string speed;  // speed name (for "speed"), else ""
};

struct TrackState {
    int track_id;
    int state;   // 0 candidate, 1 confirmed, 2 lost
    float x1, y1, x2, y2;
    std::string speed;  // emitted speed (may be empty until voted)
};

// N: consecutive detections to confirm; M: consecutive misses to mark lost;
// K: consecutive same-vote frames to emit a speed value.
class Tracker {
  public:
    // assoc_gate: center-distance association gate, in units of object size
    // (center may move up to assoc_gate * max(track_w, track_h) between frames).
    Tracker(int N = 3, int M = 3, int K = 3, float assoc_gate = 2.0f,
            int other_id = 27);

    std::vector<TrackEvent> update(const std::vector<DetResult>& dets, int frame_idx);

    int num_tracks() const { return tracks_.size(); }

    // Snapshot of current (non-deleted) tracks, for publishing.
    std::vector<TrackState> get_tracks() const;

  private:
    struct Track {
        int id = -1;
        int state = 0;        // 0 candidate, 1 confirmed, 2 lost
        int det_streak = 0;   // consecutive detections (towards N)
        int miss_streak = 0;  // consecutive misses (towards M)
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        int vote_val = -1;       // current voting cls_id
        int vote_streak = 0;     // consecutive same vote (towards K)
        std::string emitted;     // last emitted speed (avoid re-emit)
    };

    float dist_score(const Track& t, const DetResult& d) const;

    std::unordered_map<int, Track> tracks_;
    int next_id_ = 0;
    int N_, M_, K_;
    float assoc_gate_;
    int other_id_;
};

}  // namespace tsr
