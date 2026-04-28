import json
import numpy as np

with open('output/mapped_predictions.json') as f:
    pred = json.load(f)
with open('data/annotations/unav100_annotations.json') as f:
    gt = json.load(f)

# 비디오별로 GT label과 매칭되는 pred만 비교
total_matched = 0
time_errors = []
tiou_list = []

for item in pred:
    vid = item['video_id']
    info = gt['database'].get(vid, {})
    dur = info.get('duration', 0)
    gt_anns = info.get('annotations', [])

    for ev in item['events']:
        label = ev['mapped_label']
        for ann in gt_anns:
            if ann['label'] == label:
                total_matched += 1
                gt_s, gt_e = ann['segment']
                pr_s, pr_e = ev['start'], ev['end']

                start_err = abs(pr_s - gt_s)
                end_err = abs(pr_e - gt_e)
                time_errors.append((start_err, end_err))

                inter = max(0, min(pr_e, gt_e) - max(pr_s, gt_s))
                union = max(pr_e, gt_e) - min(pr_s, gt_s)
                iou = inter / union if union > 0 else 0
                tiou_list.append(iou)
                break

start_errs = [e[0] for e in time_errors]
end_errs = [e[1] for e in time_errors]

print(f'Label 매칭된 예측 수: {total_matched}')
print(f'\n시간 오차 (초):')
print(f'  Start: mean={np.mean(start_errs):.2f}, median={np.median(start_errs):.2f}, max={max(start_errs):.2f}')
print(f'  End:   mean={np.mean(end_errs):.2f}, median={np.median(end_errs):.2f}, max={max(end_errs):.2f}')
print(f'\ntIoU 분포:')
print(f'  mean={np.mean(tiou_list):.4f}, median={np.median(tiou_list):.4f}')
print(f'  tIoU >= 0.9: {sum(1 for t in tiou_list if t >= 0.9)} ({sum(1 for t in tiou_list if t >= 0.9)/len(tiou_list)*100:.1f}%)')
print(f'  tIoU >= 0.7: {sum(1 for t in tiou_list if t >= 0.7)} ({sum(1 for t in tiou_list if t >= 0.7)/len(tiou_list)*100:.1f}%)')
print(f'  tIoU >= 0.5: {sum(1 for t in tiou_list if t >= 0.5)} ({sum(1 for t in tiou_list if t >= 0.5)/len(tiou_list)*100:.1f}%)')
print(f'  tIoU >= 0.3: {sum(1 for t in tiou_list if t >= 0.3)} ({sum(1 for t in tiou_list if t >= 0.3)/len(tiou_list)*100:.1f}%)')
print(f'  tIoU < 0.1:  {sum(1 for t in tiou_list if t < 0.1)} ({sum(1 for t in tiou_list if t < 0.1)/len(tiou_list)*100:.1f}%)')

# 전체 구간(00~99) 예측 비율
full_range = 0
total_events = 0
for item in pred:
    vid = item['video_id']
    dur = gt['database'][vid]['duration']
    for ev in item['events']:
        total_events += 1
        if ev['end'] - ev['start'] > 0.95 * dur:
            full_range += 1

print(f'\n전체 구간(00~99) 예측 비율: {full_range}/{total_events} ({full_range/total_events*100:.1f}%)')
