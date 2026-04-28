"""
AVicuna prediction → CCNet ANETdetection으로 mAP 계산

Usage:
    cd /workspace/jsy/Team4/AVicuna-main
    python eval_with_ccnet.py \
        --pred output/json_prompt_full.json \
        --gt   data/annotations/unav100_annotations.json \
        --ccnet_root /workspace/jsy/Team4/CCNet-AAAI2025

기존 eval_unav100_map.py 와 비교해서 수치가 달라지는지 확인하는 용도.
"""

import argparse, json, sys
import numpy as np
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred',       required=True,  help='output/json_prompt_full.json')
    p.add_argument('--gt',         required=True,  help='unav100_annotations.json')
    p.add_argument('--ccnet_root', required=True,  help='CCNet-AAAI2025 root dir')
    p.add_argument('--subset',     default='test')
    return p.parse_args()


def main():
    args = parse_args()

    # ── CCNet ANETdetection 로드 ──────────────────────────────────
    sys.path.insert(0, args.ccnet_root)
    try:
        from libs.utils import ANETdetection
        print("ANETdetection loaded from CCNet")
    except ImportError as e:
        print(f"CCNet import 실패: {e}")
        print("libs/utils/__init__.py 또는 경로 확인 필요")
        sys.exit(1)

    # ── prediction 파일 로드 ──────────────────────────────────────
    with open(args.pred) as f:
        pred_data = json.load(f)
    print(f"Predictions: {len(pred_data)} videos")

    # ── CCNet ANETdetection이 기대하는 포맷으로 변환 ──────────────
    # CCNet ANETdetection은 내부적으로 pandas DataFrame을 쓰고
    # prediction을 dict of list로 받거나, json 파일 경로로 받음.
    # UnAV baseline과 같은 포맷: 
    # { "results": { video_id: [ {label, segment, score} ] } }
    results_dict = {}
    for item in pred_data:
        vid = item['video_id']
        results_dict[vid] = []
        for ev in item['events']:
            results_dict[vid].append({
                'label':   ev['mapped_label'],
                'segment': [ev['start'], ev['end']],
                'score':   float(ev.get('score', 1.0)),
            })

    # 임시 json 저장
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(
        suffix='.json', delete=False, mode='w', encoding='utf-8'
    )
    json.dump({'results': results_dict}, tmp, ensure_ascii=False)
    tmp.close()
    print(f"Temp prediction file: {tmp.name}")

    # ── ANETdetection 실행 ────────────────────────────────────────
    tiou_thresholds = np.linspace(0.1, 0.9, 9)

    try:
        det = ANETdetection(
            ground_truth_filename=args.gt,
            prediction_filename=tmp.name,
            tiou_thresholds=tiou_thresholds,
            subset=args.subset,
            verbose=True,
            check_status=False,
        )
        mAP, avg_mAP = det.evaluate()
    except Exception as e:
        print(f"\nANETdetection.evaluate() 실패: {e}")
        print("ANETdetection signature 확인 필요 — CCNet/libs/utils/metrics.py 열어보세요")
        os.unlink(tmp.name)
        sys.exit(1)

    os.unlink(tmp.name)

    # ── 결과 출력 ─────────────────────────────────────────────────
    paper_ref = {0.5: 60.0, 0.6: 50.4, 0.7: 49.6, 0.8: 43.5, 0.9: 36.5}

    print("\n" + "="*52)
    print("  CCNet ANETdetection — AVicuna prediction mAP")
    print("="*52)
    print(f"{'tIoU':>6}  {'CCNet eval':>10}  {'Paper':>8}  {'Diff':>8}")
    print("-"*42)

    mAP_list = mAP if hasattr(mAP, '__iter__') else [mAP]
    for i, thr in enumerate(tiou_thresholds):
        val = mAP_list[i] * 100 if i < len(mAP_list) else 0.0
        paper = paper_ref.get(round(thr, 1), '-')
        diff  = f"{val - paper:+.2f}" if isinstance(paper, float) else '-'
        print(f"  {thr:.1f}    {val:>9.2f}   {str(paper):>7}  {diff:>8}")

    print("-"*42)
    print(f"  Avg[0.1:0.9]  {avg_mAP*100:>9.2f}   {'60.3':>7}")
    print("="*52)


if __name__ == '__main__':
    main()
