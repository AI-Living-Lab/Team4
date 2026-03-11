## 📁 UnAV-100 Dataset

### 🎬 Raw Video
- [Download Raw Videos](https://drive.google.com/drive/folders/1YtKugZrNJ8iCEtncyMCPdQ1Qdrfal9_U)

### 🔎 Features Extracted with I3D + VGGish
- [Download Features](https://drive.google.com/drive/folders/1xcNnXLVfd7cJEoGUvJYHerCXQnKFoPS-)

### 🧠 Features Extracted with ONE-PEACE
- [Download Features](https://drive.google.com/drive/folders/1wKnNlNU1FHiw3lN7kaT09frlM0bg4XjI)

- Files included:
  - `av_features.tar.gz` — Video features  
  - `av_features_audio_al_retrieval.tar.gz` — Audio features

## ⚙️ video-SALMONN-2 Setup

Repository reference:  
- [video-SALMONN-2 GitHub](https://github.com/bytedance/video-SALMONN-2?tab=readme-ov-file)

### 📥 Download Checkpoints

1. Download **LLaVA-OneVision model** from [huggingface](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)


2. Download **video-SALMONN-2 model** from [huggingface](https://huggingface.co/tsinghua-ee/video-SALMONN-2)

### 📂 Expected Directory Structure
```bash
video-SALMONN-2/
├── checkpoints/
│   ├── llava_onevision_qwen2_7b_ov/
│   └── video_salmonn2_hf/
├── data/
├── llava/
├── scripts/
└── ...
