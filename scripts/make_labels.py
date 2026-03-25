import json

input_path = "data/unav100_annotations.json"
output_path = "data/unav100_class_labels.txt"

labels = set()

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

database = data["database"]

for video_id, info in database.items():
    anns = info.get("annotations", [])
    for ann in anns:
        label = ann.get("label")
        if label:
            labels.add(label.strip())

labels = sorted(labels)

with open(output_path, "w", encoding="utf-8") as f:
    for label in labels:
        f.write(label + "\n")

print("num classes:", len(labels))
print(labels[:20])