#!/bin/bash
INPUT=/home/aix23102/audiolm/vS2_eunji/data/internvid_splits.txt
OUTDIR=/data0/aix23102/internvid

download_one() {
    vid_id=$1
    start=$2
    end=$3
    outfile="$OUTDIR/${vid_id}.mp4"
    if [ -f "$outfile" ]; then
        return
    fi
    yt-dlp \
        --download-sections "*${start}-${end}" \
        --format "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
        --output "$outfile" \
        --cookies "/home/aix23102/audiolm/vS2_eunji/www.youtube.com_cookies (1).txt" \
        --no-warnings \
        --quiet \
        "https://www.youtube.com/watch?v=${vid_id}"
}
export -f download_one
export OUTDIR

cat $INPUT | xargs -P 2 -L 1 bash -c 'download_one $1 $2 $3' _
