"""
Verify time conversion roundtrip:
  raw output (relative 0-99) → absolute (seconds) → back to relative (0-99)
  Check if roundtrip matches original values
"""
import json
import re
import numpy as np

with open('data/annotations/unav100_annotations.json') as f:
    gt = json.load(f)

with open('output/json_prompt_full.json') as f:
    pred = json.load(f)

total_events = 0
roundtrip_match = 0
roundtrip_mismatch = 0
max_error = 0
errors = []

for item in pred:
    vid = item['video_id']
    dur = gt['database'][vid]['duration']
    raw = item.get('raw_output', '')

    # Extract original relative times from raw output
    raw_times = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', raw, re.IGNORECASE)

    for i, ev in enumerate(item['events']):
        if i >= len(raw_times):
            break
        total_events += 1

        # Original relative time (from raw output)
        orig_rel_start = float(raw_times[i][0])
        orig_rel_end = float(raw_times[i][1])

        # Absolute time (stored in events, already converted)
        abs_start = ev['start']
        abs_end = ev['end']

        # Convert back to relative
        if dur > 0:
            back_rel_start = abs_start / dur * 100
            back_rel_end = abs_end / dur * 100
        else:
            back_rel_start = 0
            back_rel_end = 0

        # Compare
        start_err = abs(back_rel_start - orig_rel_start)
        end_err = abs(back_rel_end - orig_rel_end)
        max_err = max(start_err, end_err)
        max_error = max(max_error, max_err)

        if max_err < 0.01:
            roundtrip_match += 1
        else:
            roundtrip_mismatch += 1
            if len(errors) < 5:
                errors.append({
                    'vid': vid,
                    'dur': dur,
                    'orig_rel': (orig_rel_start, orig_rel_end),
                    'abs': (abs_start, abs_end),
                    'back_rel': (back_rel_start, back_rel_end),
                    'error': max_err,
                })

print(f'=== Time Conversion Roundtrip Verification ===')
print(f'Total events checked: {total_events}')
print(f'Roundtrip match (<0.01 error): {roundtrip_match} ({roundtrip_match/total_events*100:.1f}%)')
print(f'Roundtrip mismatch: {roundtrip_mismatch}')
print(f'Max error: {max_error:.6f}')

# Show sample roundtrips
print(f'\n=== Sample Roundtrips (first 5 events) ===')
item = pred[0]
vid = item['video_id']
dur = gt['database'][vid]['duration']
raw = item.get('raw_output', '')
raw_times = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', raw, re.IGNORECASE)

print(f'Video: {vid} (duration: {dur}s)')
for i, ev in enumerate(item['events'][:5]):
    if i >= len(raw_times):
        break
    orig_s, orig_e = float(raw_times[i][0]), float(raw_times[i][1])
    abs_s, abs_e = ev['start'], ev['end']
    back_s, back_e = abs_s / dur * 100, abs_e / dur * 100
    match = '✅' if abs(back_s - orig_s) < 0.01 and abs(back_e - orig_e) < 0.01 else '❌'
    print(f'  Relative: {orig_s} ~ {orig_e} → Absolute: {abs_s:.2f} ~ {abs_e:.2f}s → Back: {back_s:.2f} ~ {back_e:.2f} {match}')

if errors:
    print(f'\n=== Mismatch Examples ===')
    for e in errors:
        print(f'  {e["vid"]} (dur={e["dur"]}): rel {e["orig_rel"]} → abs {e["abs"]} → back {e["back_rel"]} (err={e["error"]:.4f})')
