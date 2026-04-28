import json, numpy as np
from collections import defaultdict

def tiou(s1, s2):
    inter = max(0, min(s1[1],s2[1]) - max(s1[0],s2[0]))
    union = max(s1[1],s2[1]) - min(s1[0],s2[0])
    return inter/union if union > 0 else 0

def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1])

with open('data/annotations/unav100_annotations.json') as f:
    gt_data = json.load(f)
with open('output/label_pred_v3_10.json') as f:
    pred_data = json.load(f)

pred_vids = set(item['video_id'] for item in pred_data)

gt_by_label = defaultdict(lambda: defaultdict(list))
for vid, info in gt_data['database'].items():
    if vid not in pred_vids:
        continue
    for a in info['annotations']:
        gt_by_label[a['label']][vid].append({
            'start': a['segment'][0],
            'end': a['segment'][1]
        })

pred_by_label = defaultdict(list)
for item in pred_data:
    for ev in item['events']:
        pred_by_label[ev['mapped_label']].append({
            'video_id': item['video_id'],
            'start': ev['start'],
            'end': ev['end'],
            'score': ev['score']
        })

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = {}

for thr in thresholds:
    aps = []
    for label in sorted(gt_by_label.keys()):
        preds = sorted(pred_by_label.get(label, []),
                       key=lambda x: x['score'], reverse=True)
        gt_vids = gt_by_label[label]
        npos = sum(len(v) for v in gt_vids.values())
        if npos == 0:
            continue
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_used = {
            v: [{'s': g['start'], 'e': g['end'], 'u': False} for g in gs]
            for v, gs in gt_vids.items()
        }
        for i, p in enumerate(preds):
            vid = p['video_id']
            if vid not in gt_used:
                fp[i] = 1
                continue
            best_iou, best_j = -1, -1
            for j, g in enumerate(gt_used[vid]):
                iou = tiou([p['start'], p['end']], [g['s'], g['e']])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= thr and not gt_used[vid][best_j]['u']:
                tp[i] = 1
                gt_used[vid][best_j]['u'] = True
            else:
                fp[i] = 1
        tc = np.cumsum(tp)
        fc = np.cumsum(fp)
        rec = tc / max(npos, 1e-10)
        prec = tc / np.maximum(tc + fc, 1e-10)
        aps.append(compute_ap(rec, prec))
    results[thr] = np.mean(aps) * 100 if aps else 0

print('=== v3: NO filter + confidence + NMS + short filter ===')
for t in thresholds:
    print(f'tIoU {t:.1f}: {results[t]:.2f}')
avg = np.mean([results[t] for t in thresholds])
avg59 = np.mean([results[t] for t in [0.5,0.6,0.7,0.8,0.9]])
print(f'Avg [0.1:0.9]: {avg:.2f} (paper: 60.3)')
print(f'Avg [0.5:0.9]: {avg59:.2f}')
print(f'\nComparison:')
print(f'  Exp A (describe):      14.97')
print(f'  Exp B (label, no NO):   3.68')
print(f'  Exp C (label + NO):    31.65')
print(f'  Exp D (+ conf/NMS):    {avg:.2f}')
