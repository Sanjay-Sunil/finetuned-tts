"""Modal pipeline: data preparation, alignment, tokenizer, training and sampling.

Everything persists on the `hinglish-tts` volume (mounted at /vol):
    /vol/data/<source>/      per-dataset FLAC audio and manifests
    /vol/data/mix/           combined aligned manifests + tokenizer
    /vol/runs/<run>/         checkpoints, samples, progress.jsonl

Usage:
    modal run modal_app.py::prepare --repo SPRINGLab/IndicTTS-Hindi --source indictts_hi --speaker-col gender
    modal run modal_app.py::align --source indictts_hi --model ai4bharat/indicwav2vec-hindi
    modal run modal_app.py::build_mix --sources indictts_hi,spicor_en
    modal run --detach modal_app.py::train --config hinglish_24l.yaml
    modal run modal_app.py::generate --run hinglish_24l --step 4000
"""

import json
import os
import subprocess
from pathlib import Path

import modal

POCKET_TTS_COMMIT = "2e117a1e7df06323c9df0315c671f3cbc3b8d9fc"
POCKET = "/opt/pocket-tts"
PY = f"{POCKET}/.venv/bin/python"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("uv")
    .run_commands(
        f"git clone https://github.com/kyutai-labs/pocket-tts {POCKET}",
        f"cd {POCKET} && git checkout {POCKET_TTS_COMMIT} && uv sync --frozen",
        f"uv pip install --python {PY} pyarrow",
    )
    .env({"PYTHONPATH": "/root", "HF_HOME": "/vol/hf_cache", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("hinglish_tts")
    .add_local_dir("configs", "/root/configs")
)

app = modal.App("hinglish-pocket-tts", image=image)
vol = modal.Volume.from_name("hinglish-tts", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
common = dict(volumes={"/vol": vol}, secrets=[hf_secret])


def run(cmd: list[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=POCKET, check=True)


@app.function(cpu=16, memory=32768, timeout=3 * 3600, **common)
def prepare(
    repo: str,
    source: str,
    prefix: str = "",
    text_col: str = "text",
    speaker_col: str = "",
    max_hours: float = 1e9,
):
    cmd = [PY, "-m", "hinglish_tts.prepare_data", repo, f"/vol/data/{source}", "--source", source]
    cmd += ["--prefix", prefix, "--text-col", text_col, "--max-hours", str(max_hours)]
    cmd += ["--workers", "14"]
    if speaker_col:
        cmd += ["--speaker-col", speaker_col]
    run(cmd)
    vol.commit()


@app.function(gpu="L40S", timeout=2 * 3600, **common)
def align(source: str, model: str):
    d = f"/vol/data/{source}"
    for split in ("train", "valid"):
        run([PY, "-m", "training.scripts.align_data", f"{d}/{split}.jsonl",
             f"{d}/{split}_aligned.jsonl", "--model", model, "--device", "cuda"])
    for split in ("train", "valid"):
        n_in = sum(1 for _ in open(f"{d}/{split}.jsonl"))
        n_out = sum(1 for _ in open(f"{d}/{split}_aligned.jsonl"))
        print(f"{source} {split}: aligned {n_out}/{n_in}")
    vol.commit()


@app.function(cpu=8, timeout=3600, **common)
def build_mix(sources: str, vocab_size: int = 4000):
    """Concatenate the aligned manifests of `sources` and fit the tokenizer on them."""
    out = Path("/vol/data/mix")
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        with open(out / f"{split}_aligned.jsonl", "w") as f:
            for s in sources.split(","):
                for line in open(f"/vol/data/{s}/{split}_aligned.jsonl"):
                    f.write(line)
    # Fit on both splits' text so every character seen at validation is covered.
    run([PY, "-m", "training.scripts.train_tokenizer", str(out / "tokenizer"),
         str(out / "train_aligned.jsonl"), str(out / "valid_aligned.jsonl"),
         "--vocab-size", str(vocab_size), "--character-coverage", "1.0"])
    vol.commit()


@app.function(gpu="H100", timeout=6 * 3600, **common)
def train(config: str = "hinglish_24l.yaml", max_steps: int = 0, run_name: str = ""):
    import yaml

    cfg = yaml.safe_load(open(f"/root/configs/{config}"))
    if max_steps:
        cfg["max_steps"] = max_steps
    if run_name:
        cfg["run_dir"] = f"/vol/runs/{run_name}"
    Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(cfg["run_dir"]) / "train_config.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), allow_unicode=True)
    # Commit the volume every few minutes so checkpoints are visible while training runs.
    proc = subprocess.Popen([PY, "training/train.py", str(cfg_path)], cwd=POCKET)
    while proc.poll() is None:
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            vol.commit()
    vol.commit()
    if proc.returncode:
        raise RuntimeError(f"training exited with {proc.returncode}")


@app.function(cpu=2, timeout=600, **common)
def progress(run_name: str = "hinglish_24l", last: int = 15):
    vol.reload()
    lines = open(f"/vol/runs/{run_name}/progress.jsonl").read().splitlines()
    for line in lines[-last:]:
        print(line)
    print(sorted(os.listdir(f"/vol/runs/{run_name}")))
