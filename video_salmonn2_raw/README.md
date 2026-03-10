---
license: apache-2.0
task_categories:
- video-text-to-text
language:
- en
---
# video-SALMONN 2 Benchmark

[Github Link](https://github.com/bytedance/video-SALMONN-2)

[Paper Link](https://arxiv.org/abs/2506.15220)

- Generate the caption corresponding to the video and the audio with `video_salmonn2_test.json`
- Organize your results in the format like the following example: 

```json
[
    {
        "id": ["0.mp4"], 
        "pred": "Generated Caption"
    }
]
```

- Replace `res_file` in `eval.py` with your result file.
- Run `python3 eval.py`