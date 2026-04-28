import json
import re

with open('output/raw_predictions.json') as f:
    raw = json.load(f)
with open('output/parsed_predictions.json') as f:
    parsed = json.load(f)
with open('data/annotations/unav100_annotations.json') as f:
    gt = json.load(f)

total = len(raw)
parse_success = 0
parse_fail = 0
time_mismatch = 0
duration_mismatch = 0
fail_examples = []

for r, p in zip(raw, parsed):
    vid = r['video_id']
    raw_text = r['raw_output']
    events = p['events']
    dur = gt['database'][vid]['duration']

    # 1. Raw에서 시간 패턴 직접 추출
    # Pattern 1: "From XX to YY, label"
    p1 = re.findall(r'[Ff]rom\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*,\s*(.+?)(?=\s+[Ff]rom\s+\d|$)', raw_text)
    # Pattern 2: "Label, from XX to YY"
    p2 = re.findall(r'([A-Z][^,\n]+?),?\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', raw_text)

    raw_count = len(p1) + len(p2)
    parsed_count = len(events)

    if raw_count == 0 and parsed_count == 0:
        parse_success += 1
    elif raw_count == parsed_count:
        parse_success += 1
    elif parsed_count > 0 and raw_count > 0:
        # 개수가 다를 수 있음 (중복 제거 등)
        parse_success += 1
    else:
        parse_fail += 1
        if len(fail_examples) < 10:
            fail_examples.append({
                'vid': vid,
                'raw': raw_text[:200],
                'raw_count': raw_count,
                'parsed_count': parsed_count
            })

    # 2. 시간 역변환 검증
    for ev in events:
        # 역변환된 시간이 duration 범위 내인지
        if ev['end'] > dur * 1.05:  # 5% 여유
            duration_mismatch += 1

        # start < end 인지
        if ev['start'] > ev['end']:
            time_mismatch += 1

total_events = sum(len(p['events']) for p in parsed)

print(f'=== 파싱 검증 결과 ===')
print(f'전체 비디오: {total}')
print(f'파싱 성공: {parse_success} ({parse_success/total*100:.1f}%)')
print(f'파싱 불일치: {parse_fail} ({parse_fail/total*100:.1f}%)')
print(f'')
print(f'전체 이벤트: {total_events}')
print(f'start > end: {time_mismatch}')
print(f'end > duration: {duration_mismatch}')
print(f'')

if fail_examples:
    print(f'파싱 불일치 예시 (최대 10개):')
    for ex in fail_examples:
        print(f'  {ex["vid"]}: raw에서 {ex["raw_count"]}개, parsed {ex["parsed_count"]}개')
        print(f'    Raw: {ex["raw"][:150]}')
        print()

# 역변환 정확도 샘플 확인
print(f'=== 역변환 샘플 확인 (5개) ===')
for r, p in zip(raw[:5], parsed[:5]):
    vid = r['video_id']
    dur = gt['database'][vid]['duration']
    raw_text = r['raw_output']

    # raw에서 숫자 직접 추출
    raw_times = re.findall(r'[Ff]rom\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', raw_text)

    print(f'Video: {vid} (duration: {dur}s)')
    print(f'  Raw output: {raw_text[:120]}')
    for i, ev in enumerate(p['events'][:3]):
        if i < len(raw_times):
            raw_s, raw_e = float(raw_times[i][0]), float(raw_times[i][1])
            expected_s = raw_s / 100.0 * dur
            expected_e = raw_e / 100.0 * dur
            match_s = abs(ev['start'] - expected_s) < 0.01
            match_e = abs(ev['end'] - expected_e) < 0.01
            print(f'  Event {i}: raw=[{raw_s},{raw_e}] → expected=[{expected_s:.2f},{expected_e:.2f}] → parsed=[{ev["start"]:.2f},{ev["end"]:.2f}] {"✅" if match_s and match_e else "❌"}')
    print('---')
