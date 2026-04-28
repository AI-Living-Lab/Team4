#!/bin/bash
# Extract 48kHz mono wav from each LongVALE mp4.
# Output matches /workspace/datasets/LongVALE/audios/<vid>.wav

set -u

VIDEO_DIR="/workspace/datasets/LongVALE/videos"
AUDIO_DIR="/workspace/datasets/LongVALE/audios"
LOG_DIR="/workspace/jsy/scripts"
LOG="${LOG_DIR}/extract_wav_longvale.log"
ERR_LIST="${LOG_DIR}/extract_wav_longvale.errors.txt"

mkdir -p "${AUDIO_DIR}"
: > "${LOG}"
: > "${ERR_LIST}"

total=$(ls "${VIDEO_DIR}" | grep -c '\.mp4$')
echo "Total videos: ${total}" | tee -a "${LOG}"

done=0
ok=0
fail=0
skip=0
noaudio=0

for mp4 in "${VIDEO_DIR}"/*.mp4; do
    vid=$(basename "${mp4}" .mp4)
    wav="${AUDIO_DIR}/${vid}.wav"
    done=$((done + 1))

    if [[ -s "${wav}" ]]; then
        skip=$((skip + 1))
        continue
    fi

    # Check audio stream exists
    has_audio=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_type \
        -of default=noprint_wrappers=1:nokey=1 "${mp4}" 2>/dev/null)
    if [[ "${has_audio}" != "audio" ]]; then
        noaudio=$((noaudio + 1))
        echo "${vid} NO_AUDIO" >> "${ERR_LIST}"
        continue
    fi

    if ffmpeg -y -loglevel error -i "${mp4}" -vn -ac 1 -ar 48000 -acodec pcm_s16le \
        "${wav}" 2>> "${LOG}"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        rm -f "${wav}"
        echo "${vid} FFMPEG_FAIL" >> "${ERR_LIST}"
    fi

    if (( done % 50 == 0 )); then
        echo "[progress] ${done}/${total}  ok=${ok} skip=${skip} noaudio=${noaudio} fail=${fail}" | tee -a "${LOG}"
    fi
done

echo "===== DONE =====" | tee -a "${LOG}"
echo "total=${done} ok=${ok} skip=${skip} noaudio=${noaudio} fail=${fail}" | tee -a "${LOG}"
