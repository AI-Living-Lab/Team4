import json
from collections import Counter

with open("/home/aix23102/audiolm/vS2_eunji/eval/results/unav100_test_uf_51912/predictions.json") as f:
    preds = json.load(f)

with open("/home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json") as f:
    db = json.load(f)["database"]

gt_labels = set()
for item in db.values():
    for ann in item.get("annotations", []):
        gt_labels.add(ann["label"])

pred_labels = []
for dets in preds.values():
    for d in dets:
        pred_labels.append(d["label"])

label_counts = Counter(pred_labels)
matched   = {l: c for l, c in label_counts.items() if l in gt_labels}
unmatched = {l: c for l, c in label_counts.items() if l not in gt_labels}

print(f"전체 pred:           {len(pred_labels)}")
print(f"GT label로 매핑된 것: {sum(matched.values())} ({len(matched)} 종류)")
print(f"매핑 실패:            {sum(unmatched.values())} ({len(unmatched)} 종류)")

if unmatched:
    print("\n[매핑 실패 상위 20개]")
    for l, c in sorted(unmatched.items(), key=lambda x: -x[1])[:20]:
        print(f"  {c:4d}  '{l}'")
else:
    print("\n✅ 모든 pred가 GT label로 매핑됨")