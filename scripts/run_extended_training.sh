#!/usr/bin/env bash
# Unattended schedule for the laptop run:
#   1. wait for the running 3,000-step job to exit,
#   2. resume the same run to 6,000 steps (configs/hinglish_24l_local_6k.yaml),
#   3. meanwhile, every 1,000-step checkpoint is hard-linked into keep/ (so the
#      trainer's own rotation cannot delete it) and scored with eval_cer.
# Results: outputs/eval/results.json and outputs/eval/summary.tsv.
# Sleep and lid-close suspend are inhibited for the whole schedule.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
RUN=runs/hinglish_24l_local
PY=.venv-gpu/bin/python
LOG=runs/extended.log
mkdir -p "$RUN/keep" outputs/eval

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

evaluate() {
    local step=$1 ckpt
    ckpt=$(printf "%s/keep/checkpoint_%08d.pt" "$RUN" "$step")
    log "evaluating step $step"
    CUDA_VISIBLE_DEVICES= nice -n 10 "$PY" -m hinglish_tts.eval_cer \
        --voice voices/indictts_hi_female.wav --per-source 10 --out-dir outputs/eval \
        --model "step${step}_raw=configs/inference_local.yaml:${ckpt}#raw" \
        --model "step${step}_ema=configs/inference_local.yaml:${ckpt}" 2>&1 \
        | grep -E "^step" | while read -r line; do
            log "CER $line"
            echo -e "$(date '+%F %T')\t$line" >> outputs/eval/summary.tsv
        done
}

watch_and_eval() {
    local step=2000
    while [ "$step" -le 6000 ]; do
        local src
        src=$(printf "%s/checkpoint_%08d.pt" "$RUN" "$step")
        if [ -f "$src" ]; then
            sleep 60  # let the trainer finish writing it
            ln -f "$src" "$RUN/keep/" && evaluate "$step"
            step=$((step + 1000))
        elif ! pgrep -f "hinglish_tts.train_lowmem" >/dev/null && [ -f "$RUN/.done" ]; then
            break
        else
            sleep 60
        fi
    done
    log "evaluation watcher finished"
}

schedule() {
    log "waiting for the 3,000-step job to exit"
    while pgrep -f "hinglish_tts.train_lowmem ../../configs/hinglish_24l_local.yaml" >/dev/null; do
        sleep 60
    done
    log "resuming to 6,000 steps"
    (cd third_party/pocket-tts && PYTHONPATH="$ROOT" "$ROOT/$PY" -m hinglish_tts.train_lowmem \
        ../../configs/hinglish_24l_local_6k.yaml >> "$ROOT/runs/train_local_6k.log" 2>&1)
    log "training exited with $?"
    touch "$RUN/.done"
}

watch_and_eval &
schedule
wait
log "all done"
