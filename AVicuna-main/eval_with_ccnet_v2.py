"""
CCNet ANETdetection으로 AVicuna prediction mAP 계산
- label_id 매핑 처리 포함

Usage:
    cd /workspace/jsy/Team4/AVicuna-main
    python eval_with_ccnet_v2.py \
        --pred output/json_prompt_full.json \
        --gt   data/annotations/unav100_annotations.json \
        --ccnet_root /workspace/jsy/Team4/CCNet-AAAI2025
"""

import argparse, json, sys, os
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred',       required=True)
    p.add_argument('--gt',         required=True)
    p.add_argument('--ccnet_root', required=True)
    p.add_argument('--subset',     default='test')
    return p.parse_args()


def main():
    args = parse_args()

    # ── CCNet metrics.py 직접 임포트 (NMS 불필요) ─────────────────
    sys.path.insert(0, os.path.join(args.ccnet_root, 'libs', 'utils'))
    from metrics import ANETdetection

    # ── GT에서 label → label_id 매핑 구성 ────────────────────────
    with open(args.gt) as f:
        gt_data = json.load(f)

    label_to_id = {}
    for vid, info in gt_data['database'].items():
        for ann in info.get('annotations', []):
            lbl = ann['label']
            lid = int(ann['label_id'])
            label_to_id[lbl] = lid

    print(f"Label vocab size: {len(label_to_id)}")

    # ── prediction 로드 & DataFrame 변환 ─────────────────────────
    with open(args.pred) as f:
        pred_data = json.load(f)

    rows = []
    skip = 0
    for item in pred_data:
        vid = item['video_id']
        for ev in item['events']:
            lbl = ev['mapped_label']
            if lbl not in label_to_id:
                skip += 1
                continue
            rows.append({
                'video-id': vid,
                't-start':  float(ev['start']),
                't-end':    float(ev['end']),
                'label':    label_to_id[lbl],
                'score':    float(ev.get('score', 1.0)),
            })

    pred_df = pd.DataFrame(rows)
    print(f"Predictions: {len(pred_df)} rows ({skip} skipped, label not in GT vocab)")
    print(pred_df.head(3))

    # ── ANETdetection 실행 ────────────────────────────────────────
    tiou_thresholds = np.linspace(0.1, 0.9, 9)

    det = ANETdetection(
        ant_file=args.gt,
        split=args.subset,
        tiou_thresholds=tiou_thresholds,
        label='label_id',
        num_workers=4,
    )

    mAP, avg_mAP = det.evaluate(pred_df, verbose=True)

    # ── 결과 출력 ─────────────────────────────────────────────────
    paper_ref = {0.5: 60.0, 0.6: 50.4, 0.7: 49.6, 0.8: 43.5, 0.9: 36.5}

    print("\n" + "="*52)
    print("  CCNet ANETdetection — AVicuna mAP")
    print("="*52)
    print(f"{'tIoU':>6}  {'CCNet':>8}  {'Paper':>8}  {'Diff':>8}")
    print("-"*42)
    for thr, val in zip(tiou_thresholds, mAP):
        v = val * 100
        p = paper_ref.get(round(thr, 1), None)
        diff = f"{v-p:+.2f}" if p else "  -"
        print(f"  {thr:.1f}   {v:>8.2f}  {str(p or '-'):>8}  {diff:>8}")
    print("-"*42)
    print(f"  Avg[0.1:0.9]  {avg_mAP*100:.2f}  (paper: 60.3)")
    print("="*52)


if __name__ == '__main__':
    main()
