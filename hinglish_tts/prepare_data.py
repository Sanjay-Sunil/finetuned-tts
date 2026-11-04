"""Turn a Hugging Face parquet speech dataset into Pocket TTS training manifests.

Each source dataset is decoded to 24 kHz mono FLAC, its transcripts are
normalized with `text_norm.normalize`, utterances outside the duration window
are dropped, and a few utterances per speaker are held out for validation.

Output (in out_dir):
    wav/<id>.flac
    train.jsonl, valid.jsonl   {"path", "duration", "transcript", "speaker", "source"}

The manifests still need word alignments (training/scripts/align_data.py)
before Pocket TTS can train on them.
"""

import io
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import typer
from huggingface_hub import HfApi, hf_hub_download
from scipy.signal import resample_poly

from hinglish_tts.text_norm import normalize

SAMPLE_RATE = 24_000

app = typer.Typer(pretty_exceptions_show_locals=False)


def decode(audio_bytes: bytes) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wav


def process_shard(
    repo: str,
    filename: str,
    out_dir: str,
    source: str,
    text_col: str,
    audio_col: str,
    speaker_col: str | None,
    min_sec: float,
    max_sec: float,
    style_col: str | None = None,
    exclude_styles: frozenset[str] = frozenset(),
) -> list[dict]:
    local = hf_hub_download(repo, filename, repo_type="dataset")
    pf = pq.ParquetFile(local)
    wav_dir = Path(out_dir) / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source}_{Path(filename).stem}"
    rows = []
    idx = 0
    for batch in pf.iter_batches(batch_size=64):
        for rec in batch.to_pylist():
            idx += 1
            if style_col and str(rec[style_col]) in exclude_styles:
                continue
            text = normalize(str(rec[text_col] or ""))
            if not text:
                continue
            audio = rec[audio_col]
            wav = decode(audio["bytes"] if isinstance(audio, dict) else audio)
            duration = len(wav) / SAMPLE_RATE
            if not min_sec <= duration <= max_sec:
                continue
            peak = float(np.abs(wav).max())
            if peak < 1e-3:
                continue
            path = wav_dir / f"{stem}_{idx:06d}.flac"
            sf.write(path, wav / max(peak, 1.0), SAMPLE_RATE)
            speaker = str(rec[speaker_col]) if speaker_col else source
            rows.append(
                {
                    "path": str(path.resolve()),
                    "duration": round(duration, 3),
                    "transcript": text,
                    "speaker": f"{source}:{speaker}",
                    "source": source,
                }
            )
    return rows


@app.command()
def main(
    repo: Annotated[str, typer.Argument(help="HF dataset repo, e.g. SPRINGLab/IndicTTS-Hindi")],
    out_dir: Annotated[str, typer.Argument()],
    source: Annotated[str, typer.Option(help="short tag used in ids and manifests")],
    prefix: Annotated[str, typer.Option(help="only parquet files under this path")] = "",
    text_col: Annotated[str, typer.Option()] = "text",
    audio_col: Annotated[str, typer.Option()] = "audio",
    speaker_col: Annotated[str | None, typer.Option()] = None,
    max_hours: Annotated[float, typer.Option(help="stop adding shards past this")] = 1e9,
    shards: Annotated[str, typer.Option(help="comma-separated parquet names to use (default: all)")] = "",
    style_col: Annotated[str | None, typer.Option()] = None,
    exclude_styles: Annotated[str, typer.Option(help="comma-separated style values to drop")] = "",
    min_sec: Annotated[float, typer.Option()] = 1.0,
    max_sec: Annotated[float, typer.Option()] = 20.0,
    valid_per_speaker: Annotated[int, typer.Option()] = 20,
    workers: Annotated[int, typer.Option()] = 8,
    seed: Annotated[int, typer.Option()] = 0,
):
    files = sorted(
        f
        for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.endswith(".parquet") and f.startswith(prefix) and "test" not in Path(f).name
    )
    if shards:
        wanted = set(shards.split(","))
        files = [f for f in files if Path(f).name in wanted]
    # Shards are often sorted by speaker, so visit them in a fixed shuffled order
    # to keep both speakers when max_hours cuts the list short.
    random.Random(seed).shuffle(files)
    rows: list[dict] = []
    with ProcessPoolExecutor(workers) as pool:
        futures = [
            pool.submit(
                process_shard, repo, f, out_dir, source, text_col, audio_col,
                speaker_col, min_sec, max_sec, style_col,
                frozenset(filter(None, exclude_styles.split(","))),
            )
            for f in files
        ]
        for fut in futures:
            rows += fut.result()
            hours = sum(r["duration"] for r in rows) / 3600
            print(f"{source}: {len(rows)} utterances, {hours:.2f} h", flush=True)
            if hours >= max_hours:
                for other in futures:
                    other.cancel()
                break

    rng = random.Random(seed)
    rng.shuffle(rows)
    hours = 0.0
    kept = []
    for r in rows:
        if hours >= max_hours:
            break
        kept.append(r)
        hours += r["duration"] / 3600
    valid, train, per_speaker = [], [], {}
    for r in kept:
        n = per_speaker.get(r["speaker"], 0)
        if n < valid_per_speaker:
            valid.append(r)
            per_speaker[r["speaker"]] = n + 1
        else:
            train.append(r)
    out = Path(out_dir)
    for name, split in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"{source}: train {len(train)} ({sum(r['duration'] for r in train) / 3600:.2f} h), "
        f"valid {len(valid)}, speakers {sorted(per_speaker)}"
    )


if __name__ == "__main__":
    app()
