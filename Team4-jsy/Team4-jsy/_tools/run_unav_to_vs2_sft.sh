# test set 만들 때
python /home/aix23102/audiolm/vS2_eunji/_tools/unav_to_vs2_sft.py \
  --unav_json /home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json \
  --video_dir /data0/aix23102/unav_100/videos \
  --audio_dir /data0/aix23102/unav_100/audio \
  --split test \
  --out /home/aix23102/audiolm/vS2_eunji/data/unav100_test_dense.json \
  --skip_missing_files \
  --mode dense

# train SFT 만들 때 (GT 포함)
python /home/aix23102/audiolm/vS2_eunji/_tools/unav_to_vs2_sft.py \
  --unav_json /home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json \
  --video_dir /data0/aix23102/unav_100/videos \
  --audio_dir /data0/aix23102/unav_100/audio \
  --split train \
  --out /home/aix23102/audiolm/vS2_eunji/data/unav100_train_dense.json \
  --skip_missing_files \
  --mode dense

#   # single mode (기존 방식 - annotation 하나당 샘플 1개)
# python /home/aix23102/audiolm/vS2_eunji/_tools/unav_to_vs2_sft.py \
#   --unav_json /home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json \
#   --video_dir /data0/aix23102/unav_100/videos \
#   --audio_dir /data0/aix23102/unav_100/audio \
#   --split train \
#   --out /home/aix23102/audiolm/vS2_eunji/data/unav100_train_single.json \
#   --skip_missing_files \
#   --mode single