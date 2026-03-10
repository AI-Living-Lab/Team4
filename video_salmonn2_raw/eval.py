import time
import copy
import openai
import os
import json
import concurrent.futures
import threading
from tqdm import tqdm
import ast
import random
import re

LOCK = threading.Lock()

summarize_prompt_system = '''
### Task:
A good video description is one that describes the various details in the video. You task is to judge whether a video description is good or not. You will be provided all the base events in the video, and also a video description to be evaluated. You need to determine which video events are described in the given video description.

### Input Format:
- There are totally {event_num} base events in the video. All the base events in the video will be provided in List format, i.e. ["xxx", "xxx", ...]
- The video description to be evaluated will be provided as well.

### Output Format:
Given the video desciption, besides the events described correctly, there might be events that are missed, described incorrectly and hallucination. You need to determine the number of missed events, incorrect events and hallucination events. You are also required to list these events out.
You output should be in Python dictionary format:

{{"Missed": x, "Incorrect": x, "Hallucination": x, "Missed Event": [...], "Incorrect Event": [...], "Hallucination Event": [...] }}
'''

summarize_prompt_system_2 = '''
### Task:
A good video description is one that describes the various details in the video. You task is to judge whether a video description is good or not. You will be provided all the base events in the video, and also a video description to be evaluated. You need to determine which video events are described in the given video description.

### Input Format:
- There are totally {event_num} base events in the video. All the base events in the video will be provided in List format, i.e. ["xxx", "xxx", ...]
- The video description to be evaluated will be provided as well.

### Output Format:
Given the video desciption, you need to determine the number of missed events, correct events, incorrect events and hallucination events. Make sure that: "missed" + "correct" + "incorrect" = {event_num}
You output should be in Python dictionary format:

{{"Missed": x, "Correct": x, "Incorrect": x, "Hallucination": x}}
'''

summarize_prompt = '''
#### Events In The Video
{events_in_video}

#### Video Description To Be Rated
{cap_to_be_rated}

Given base events in the video and the video description, please count the missed, incorrect and hallucination events and list them out. 
'''

summarize_prompt_2 = '''
#### Events In The Video
{events_in_video}

#### Video Description To Be Rated
{cap_to_be_rated}

Given base events in the video and the video description, please count the missed, correct, incorrect and hallucination events. 
'''

seed = 2024

def gpt_caption(ref, pred, summarize_prompt_system=None, summarize_prompt=None):
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    temp_query_system = summarize_prompt_system.format(event_num=len(ref))
    temp_summarize_prompt = summarize_prompt.format(events_in_video=ref, cap_to_be_rated=pred)

    msg = [
        {"role": "system", "content": temp_query_system},
        {"role": "user", "content": temp_summarize_prompt}
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msg,
        seed=seed,
        temperature=0.0,
        top_p=0.1,
    )
    return completion.choices[0].message.content

res_file = "/path/to/your/file.json" 
output_json = "/path/to/output/file.json"

events_file = "video_salmonn2_test.json" 

with open(res_file, 'r') as fp:
    res_data = json.load(fp)

with open(events_file, 'r') as fp:
    events_data = json.load(fp)

map_dic = {}
for item in events_data:
    map_dic[item["video"]] = item
    
for item in res_data:
    if item['id'][0] in map_dic:
        events = map_dic[item['id'][0]]["events"]
        map_dic[item['id'][0]] = item
        map_dic[item['id'][0]]["events"] = events

res_data = list(map_dic.values())
print(len(res_data))

def reduce_repeated_words(text):
    pattern = "."
    for i in range(1, 50):
        p = pattern * i
        text = re.sub(f'({p})' + r'\1{4,200}', r'\1', text)
    for i in range(50, 100):
        p = pattern * i
        text = re.sub(f'({p})' + r'\1{3,200}', r'\1', text)
    return text

def gpt_extract(item):
    try:
        if isinstance(item['pred'], list):
            item['pred'] = item['pred'][0]
        if "<|im_end|>" not in item['pred']:
            text = reduce_repeated_words(item["pred"])
        else:
            text = item['pred'].replace("<|im_end|>", "")
        res = gpt_caption(item['events'], text, summarize_prompt_system=summarize_prompt_system, summarize_prompt=summarize_prompt)

        miss = int(res.split('"Missed":')[1].split('"Incorrect"')[0].strip().replace(",", ""))
        incor = int(res.split('"Incorrect":')[1].split('"Hallucination"')[0].strip().replace(",", ""))
        hall = int(res.split('"Hallucination":')[1].split('"Missed Event"')[0].strip().replace(",", ""))

        try:
            miss_event = json.loads(res.split('"Missed Event":')[1].split('"Incorrect Event"')[0].strip()[:-1])
        except Exception as e:
            miss_event = eval(res.split('"Missed Event":')[1].split('"Incorrect Event"')[0].strip()[:-1])
        
        try:
            incor_event = json.loads(res.split('"Incorrect Event":')[1].split('"Hallucination Event"')[0].strip()[:-1])
        except Exception as e:
            incor_event = eval(res.split('"Incorrect Event":')[1].split('"Hallucination Event"')[0].strip()[:-1])
        
        try:
            hall_event = json.loads(res.split('"Hallucination Event":')[1].split('}')[0].strip())
        except Exception as e:
            hall_event = eval(res.split('"Hallucination Event":')[1].split('}')[0].strip())

        item["Missed"] = miss
        item["Incorrect"] = incor
        item["Hallucination"] = hall
        item["Missed Event"] = miss_event
        item["Incorrect Event"] = incor_event
        item["Hallucination Event"] = hall_event

        return item

    except Exception as e:
        return item

def gpt_extract_2(item):
    try:
        if "<|im_end|>" not in item['pred']:
            text = reduce_repeated_words(item["pred"])
        else:
            text = item['pred'].replace("<|im_end|>", "")
        res = gpt_caption(item['events'], text, summarize_prompt_system=summarize_prompt_system_2, summarize_prompt=summarize_prompt_2)

        miss = int(res.split('"Missed":')[1].split('"Correct"')[0].strip().replace(",", ""))
        cor = int(res.split('"Correct":')[1].split('"Incorrect"')[0].strip().replace(",", ""))
        incor = int(res.split('"Incorrect":')[1].split('"Hallucination"')[0].strip().replace(",", ""))
        hall = int(res.split('"Hallucination":')[1].split('}')[0].strip().replace(",", ""))

        assert miss + cor + incor == len(item["events"])

        item["Missed"] = miss
        item["Incorrect"] = incor
        item["Hallucination"] = hall
        item["Correct"] = cor

        return item

    except Exception as e:
        return item

total_result = []

for i in range(7):

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        responses = list(tqdm(executor.map(gpt_extract, res_data)))

    seed += 1
    result = [r for r in responses if "Hallucination" in r]
    ignore = [r for r in responses if "Hallucination" not in r]
    try:
        with open(output_json, 'w') as fp:
            json.dump(result, fp, indent=4, ensure_ascii=False)
    except Exception as e:
        print(e)
    print(output_json)
    print(len(result), len(ignore))

    k = 0

    while len(ignore) != 0 and k < 8:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(100, len(ignore))) as executor:
            responses = list(tqdm(executor.map(gpt_extract, ignore)))

        seed += 1
        result += [r for r in responses if "Hallucination" in r]
        ignore = [r for r in responses if "Hallucination" not in r]
        try:
            with open(output_json, 'w') as fp:
                json.dump(result, fp, indent=4)
        except Exception as e:
            print(e)
        print(output_json)
        print(len(result), len(ignore))

        k += 1

    if len(ignore) > 0:
        k = 0
        print("Version 2 GPT request")
        while len(ignore) != 0 and k < 3:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                responses = list(tqdm(executor.map(gpt_extract_2, ignore)))
            
            seed += 1
            result += [r for r in responses if "Hallucination" in r]
            ignore = [r for r in responses if "Hallucination" not in r]
            try:
                with open(output_json, 'w') as fp:
                    json.dump(result, fp, indent=4)
            except Exception as e:
                print(e)
            print(output_json)
            print(len(result), len(ignore))
            k += 1

    eer = sum([it['Missed'] + it['Incorrect'] + it['Hallucination'] for it in result]) / sum([len(it["events"]) for it in result])
    left_r = sum([it['Missed'] for it in result]) / sum([len(it["events"]) for it in result])
    inc_r = sum([it['Incorrect'] for it in result]) / sum([len(it["events"]) for it in result])
    fan_r = sum([it['Hallucination'] for it in result]) / sum([len(it["events"]) for it in result])

    print(f"EER: {eer}, MISS: {left_r}, INCORRECT: {inc_r}, HALLUCINATION: {fan_r}")
    print(f"{eer:.3f}, {left_r:.3f}, {inc_r:.3f}, {fan_r:.3f}")

    total_result.append([eer, left_r, inc_r, fan_r])


sorted_result = sorted(total_result, key=lambda x: x[0])
print(f"EER: {sorted_result[3][0]:.3f}, MISS: {sorted_result[3][1]:.3f}, INCORRECT: {sorted_result[3][2]:.3f}, HALLUCINATION: {sorted_result[3][3]:.3f}")