#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watch_wandb.py — 돌고 있는 GDPO 런이 안정적인지 wandb 에서 훑어본다.

사용:
  source paths.env
  python3 gdpo/watch_wandb.py                       # 기본 = noaudio 런
  python3 gdpo/watch_wandb.py <run_name> [최근 N]

보는 것 (불안정 신호에 ⚠️ 표시):
  loss / grad_norm      — nan·inf, grad_norm 급등(>10) 
  rewards/<채널>        — format·count·global·local·precision 5채널 추세
  completions/mean_len  — 붕괴(너무 짧아짐) 감지
  val_*/combined        — val 곡선 (200 step 마다)
"""
import os
import sys

DEFAULT_RUN = "sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_noaudio"
ENTITY = os.environ.get("WANDB_ENTITY", "guma017-ewha-womans-university")


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    import wandb
    api = wandb.Api()
    # ⚠️ api.runs() 는 "entity/project" 를 요구한다 (entity 만 주면 project 없다고 실패).
    #    WANDB_PROJECT 가 안 잡혀 있으면 entity 의 project 를 전부 훑어 run 을 찾는다.
    projects = ([os.environ["WANDB_PROJECT"]] if os.environ.get("WANDB_PROJECT")
                else [pr.name for pr in api.projects(ENTITY)])
    runs = []
    for proj in projects:
        try:
            runs += [r for r in api.runs(f"{ENTITY}/{proj}") if r.name == run_name]
        except Exception:
            continue
    if not runs:
        print(f"run '{run_name}' 없음 (entity={ENTITY}, projects={projects}).")
        return 1
    run = sorted(runs, key=lambda r: r.created_at)[-1]
    print(f"run={run.name}  state={run.state}  url={run.url}")

    hist = run.history(pandas=True)
    if hist is None or len(hist) == 0:
        print("아직 로깅된 step 없음.")
        return 0

    step_col = "_step" if "_step" in hist.columns else hist.columns[0]
    tail = hist.tail(last_n)

    # ⚠️ HF Trainer 는 모든 스칼라에 "train/" 접두사를 붙여 로깅한다.
    #    (train/loss, train/rewards/format, ...) — 접두사 없이 찾으면 아무것도 안 잡힌다.
    P = "train/"

    def col(name):
        return P + name if P + name in hist.columns else (name if name in hist.columns else None)

    base = [c for c in (col("loss"), col("grad_norm"), col("learning_rate"),
                        col("reward"), col("completion_length"), col("clip_frac")) if c]
    # 채널별 총합만 (데이터셋별 /unav /charades 는 아래에서 따로 요약)
    chans = sorted(c for c in hist.columns
                   if c.startswith(P + "rewards/") and c.count("/") == 2)
    watch = base + chans

    # step 별 나열은 길어서, 구간 평균 추세로 본다 (곡선 형태 파악이 목적)
    import math
    n = len(hist)
    nb = min(10, max(2, n // 25))
    size = math.ceil(n / nb)
    print(f"\n구간 평균 추세 ({n} step 을 {nb} 구간으로):")
    hdr = ["step"] + [c.split("/")[-1][:9] for c in watch]
    print("  " + "  ".join(f"{h:>9s}" for h in hdr))
    for b in range(nb):
        blk = hist.iloc[b * size:(b + 1) * size]
        if len(blk) == 0:
            continue
        cells = [f"{int(blk[step_col].iloc[-1]):>9d}"]
        for c in watch:
            v = blk[c].dropna().mean() if c in blk else float("nan")
            cells.append(f"{v:>9.4f}" if v == v else f"{'-':>9s}")
        print("  " + "  ".join(cells))

    # 데이터셋별 리워드 (최근 구간)
    print("\n데이터셋별 리워드 (최근 50 step 평균):")
    recent = hist.tail(50)
    for tag in ("unav", "charades"):
        cs = sorted(c for c in hist.columns if c.startswith(P + "rewards/") and c.endswith("/" + tag))
        if not cs:
            continue
        bits = [f"{c.split('/')[-2]}={recent[c].dropna().mean():.4f}" for c in cs
                if len(recent[c].dropna())]
        print(f"  {tag:9s} " + "  ".join(bits))

    # 불안정 신호
    print("\n진단:")
    bad = False
    for c in (col("loss"), col("grad_norm")):
        if not c or c not in hist.columns:
            continue
        s = hist[c].dropna()
        if len(s) == 0:
            continue
        if not (s == s).all():
            print(f"  ⚠️ {c} 에 NaN 존재"); bad = True
        if c.endswith("grad_norm") and (s > 10).any():
            print(f"  ⚠️ grad_norm 이 10 초과한 step 존재 (max={s.max():.2f})"); bad = True
    vcols = [c for c in hist.columns if "combined" in c]
    for c in vcols:
        s = hist[c].dropna()
        if len(s):
            print(f"  {c}: 최근 {s.iloc[-1]:.4f} / 최고 {s.max():.4f} (step {int(hist.loc[s.idxmax(), step_col])})")
    if not bad:
        print("  이상 신호 없음.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
