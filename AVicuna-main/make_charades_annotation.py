"""
Charades-STA txt → UnAV-100 형식 annotation json 변환
Usage:
    python make_charades_annotation.py
"""
import json
from pathlib import Path

VIDEO_DIR = "/workspace/datasets/charades_sta/videos"
TRAIN_TXT = "/workspace/datasets/charades_sta/annotations/charades_sta_train.txt"
TEST_TXT  = "/workspace/datasets/charades_sta/annotations/charades_sta_test.txt"
OUT_JSON  = "/workspace/jsy/Team4/AVicuna-main/data/charades_sta_annotations.json"

def parse_txt(txt_path, subset):
    entries = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta, query = line.split("##")
            parts  = meta.strip().split()
            vid_id = parts[0]
            start  = float(parts[1])
            end    = float(parts[2])

            if vid_id not in entries:
                # duration은 영상마다 다르지만 annotation에 없음
                # feature 추출용으로는 video_id + subset만 필요하므로 duration=0 으로 우선 채움
                entries[vid_id] = {
                    "subset":   subset,
                    "duration": 0.0,
                    "annotations": []
                }
            entries[vid_id]["annotations"].append({
                "segment": [start, end],
                "label":   query.strip()
            })
    return entries

print("Parsing train/test txt...")
db = {}
db.update(parse_txt(TRAIN_TXT, "train"))
db.update(parse_txt(TEST_TXT,  "test"))

# duration을 실제 mp4에서 읽기 (있으면)
try:
    import cv2
    print("Reading durations from mp4...")
    for vid_id, info in db.items():
        mp4 = Path(VIDEO_DIR) / f"{vid_id}.mp4"
        if mp4.exists():
            cap = cv2.VideoCapture(str(mp4))
            fps   = cap.get(cv2.CAP_PROP_FPS)
            nf    = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            info["duration"] = nf / fps if fps > 0 else 0.0
            cap.release()
except ImportError:
    print("cv2 없음 — duration=0으로 저장 (feature 추출에는 영향 없음)")

out = {"database": db}
import os; os.makedirs(str(Path(OUT_JSON).parent), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print(f"저장 완료: {OUT_JSON}")
print(f"  train: {sum(1 for v in db.values() if v['subset']=='train')} videos")
print(f"  test:  {sum(1 for v in db.values() if v['subset']=='test')} videos")
