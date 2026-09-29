std_msgs/Header header
uint32 track_id
float32 x1
float32 y1
float32 x2
float32 y2
float32 det_conf
float32 cls_conf
uint8 state        # 0=candidate 1=confirmed 2=lost
string speed_value # e.g. "pl20" / "il80" / "other" / "" (not yet voted)
