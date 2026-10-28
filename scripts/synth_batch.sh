#!/usr/bin/env bash
# Render and measure planned synthetic clips on a GPU box, two stages at once.
#
#   scripts/synth_batch.sh render   <prefix>   # Blender (Cycles) -> encode -> prepare
#   scripts/synth_batch.sh pipeline <prefix>   # unchanged pipeline on each render
#
# Run both, in two shells: the render loop keeps the GPU busy while the
# pipeline loop tracks and calls the NIM on what has already rendered. Each
# clip's progress is a marker file in data/synth/<clip>/ (RENDERED, DONE,
# FAILED), so either loop can be stopped and restarted. A loop exits when
# nothing is left to do, or when data/synth/STOP exists.
#
# Environment: SNOW_BLENDER (the Blender binary), SNOW_NIM_PREVIEW=0 and
# SNOW_NIM_TARGET for a self-hosted NIM, SNOW_SYNTH_SAMPLES (default 16),
# SNOW_SYNTH_S3 (optional s3:// prefix: each finished clip is copied there
# in data/'s layout, without its frames).
set -uo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?render or pipeline}
PREFIX=${2:-synthb}
SAMPLES=${SNOW_SYNTH_SAMPLES:-16}
SNOW=(.venv/bin/python -m snow.cli)
SYNTH=data/synth

log() { echo "$(date -u +%FT%TZ) $*"; }

sync_clip() {
  local name=$1
  [ -n "${SNOW_SYNTH_S3:-}" ] || return 0
  aws s3 sync --quiet --exclude 'frames/*' --exclude '*.blend1' \
    "$SYNTH/$name" "$SNOW_SYNTH_S3/data/synth/$name" || return 1
  for f in data/poses/"$name".* data/poses/"$name"-oracle.* data/clips/"$name".* \
           data/clips/"$name"-oracle.*; do
    [ -e "$f" ] || continue
    aws s3 cp --quiet "$f" "$SNOW_SYNTH_S3/$f" || return 1
  done
}

while [ ! -e "$SYNTH/STOP" ]; do
  worked=0
  for dir in "$SYNTH/$PREFIX"*/; do
    dir=${dir%/}
    name=$(basename "$dir")
    [ -e "$dir/trick.json" ] || continue
    [ -e "$dir/FAILED" ] && continue
    [ -e "$SYNTH/STOP" ] && break
    if [ "$STAGE" = render ] && [ ! -e "$dir/RENDERED" ]; then
      [ -e "$dir/RENDERING" ] && continue
      echo $$ > "$dir/RENDERING"
      t0=$(date +%s)
      log "render $name"
      if "${SNOW[@]}" synth render --only "$name" --engine cycles --samples "$SAMPLES" \
           > "$dir/render.log" 2>&1 \
         && "${SNOW[@]}" synth prepare --only "$name" >> "$dir/render.log" 2>&1; then
        rm -rf "$dir/frames" "$dir/scene.blend1"
        echo $(( $(date +%s) - t0 )) > "$dir/RENDERED"
        log "rendered $name in $(cat "$dir/RENDERED")s"
      else
        echo "render" > "$dir/FAILED"
        log "FAILED render $name"
      fi
      rm -f "$dir/RENDERING"
      worked=1
    elif [ "$STAGE" = pipeline ] && [ -e "$dir/RENDERED" ] && [ ! -e "$dir/DONE" ]; then
      t0=$(date +%s)
      log "pipeline $name"
      if "${SNOW[@]}" synth pipeline --only "$name" > "$dir/pipeline.log" 2>&1 \
         && [ -s "data/poses/$name.features.json" ]; then
        if sync_clip "$name"; then
          echo $(( $(date +%s) - t0 )) > "$dir/DONE"
          log "done $name in $(cat "$dir/DONE")s"
        else
          log "s3 sync failed for $name; will retry"
        fi
      else
        echo "pipeline" > "$dir/FAILED"
        log "FAILED pipeline $name"
      fi
      worked=1
    fi
  done
  if [ "$worked" = 0 ]; then
    if [ "$STAGE" = render ]; then break; fi
    # The pipeline loop waits for the render loop while anything is unrendered.
    pending=0
    for dir in "$SYNTH/$PREFIX"*/; do
      [ -e "$dir/trick.json" ] && [ ! -e "$dir/RENDERED" ] && [ ! -e "$dir/FAILED" ] && pending=1
    done
    [ "$pending" = 1 ] || break
    sleep 30
  fi
done
log "$STAGE loop finished"
