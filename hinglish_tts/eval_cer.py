"""Character error rate of generated speech on held-out sentences.

For each model (a released Pocket TTS language, or a config + training
checkpoint), synthesize held-out validation sentences with one fixed voice
prompt, transcribe them with a character-level wav2vec2 CTC model for the
sentence's language, and report CER per source.

Hindi and English are scored separately: an ASR model for one script cannot
fairly transcribe the other, so code-mixed sentences are left to listening.

    python -m hinglish_tts.eval_cer --voice voices/indictts_hi_female.wav \
        --model base=language:english \
        --model step2000=configs/inference_local.yaml:runs/hinglish_24l_local/checkpoint_00002000.pt
"""

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Annotated

import jiwer
import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch
import typer
from pocket_tts import TTSModel
from scipy.signal import resample_poly

from hinglish_tts.text_norm import normalize, sentence_language

ASR_MODELS = {"hi": "theainerd/Wav2Vec2-large-xlsr-hindi", "en": "facebook/wav2vec2-base-960h"}
ASR_RATE = 16_000

app = typer.Typer(pretty_exceptions_show_locals=False)


def clean(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.lower() if lang == "en" else text
    return re.sub(r"\s+", " ", text).strip()


class Transcriber:
    def __init__(self, device: str):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = device
        self.models = {}
        for lang, name in ASR_MODELS.items():
            proc = Wav2Vec2Processor.from_pretrained(name)
            model = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
            self.models[lang] = (proc, model)

    @torch.no_grad()
    def __call__(self, wav: np.ndarray, sr: int, lang: str) -> str:
        proc, model = self.models[lang]
        g = np.gcd(sr, ASR_RATE)
        wav = resample_poly(wav, ASR_RATE // g, sr // g).astype(np.float32)
        inputs = proc(wav, sampling_rate=ASR_RATE, return_tensors="pt").to(self.device)
        ids = model(**inputs).logits.argmax(-1)
        return proc.batch_decode(ids)[0]


def load_sentences(manifests: list[str], per_source: int, seed: int) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for m in manifests:
        for line in open(m):
            r = json.loads(line)
            by_source.setdefault(r["source"], []).append(r)
    rng = random.Random(seed)
    picked = []
    for source, rows in sorted(by_source.items()):
        for r in rng.sample(rows, min(per_source, len(rows))):
            picked.append({"source": source, "text": r["transcript"], "path": r["path"]})
    return picked


def load_model(spec: str) -> TTSModel:
    if spec.startswith("language:"):
        return TTSModel.load_model(language=spec.removeprefix("language:"))
    config, _, ckpt = spec.partition(":")
    use_ema = not ckpt.endswith("#raw")
    ckpt = ckpt.removesuffix("#raw")
    model = TTSModel.load_model(config=config)
    if ckpt:
        model.load_training_checkpoint(ckpt, use_ema=use_ema)
    return model


@app.command()
def main(
    voice: Annotated[str, typer.Option()],
    model: Annotated[list[str], typer.Option(help="name=language:<lang> or name=<config>:<ckpt.pt>[#raw]")],
    manifests: Annotated[list[str], typer.Option()] = [
        "data/local/mix/valid_aligned.jsonl"
    ],
    per_source: Annotated[int, typer.Option()] = 10,
    out_dir: Annotated[str, typer.Option()] = "outputs/eval",
    device: Annotated[str, typer.Option()] = "cpu",
    seed: Annotated[int, typer.Option()] = 0,
):
    sentences = load_sentences(manifests, per_source, seed)
    asr = Transcriber(device)
    results = {}
    for spec in model:
        name, _, target = spec.partition("=")
        # "reference" scores the real recordings: the ASR's own error floor.
        tts = None if target == "reference" else load_model(target)
        state = tts.get_state_for_audio_prompt(voice) if tts else None
        refs: dict[str, list[str]] = {}
        hyps: dict[str, list[str]] = {}
        rows = []
        for i, s in enumerate(sentences):
            text = normalize(s["text"])
            lang = sentence_language(text)
            if tts is None:
                audio, sr = sf.read(s["path"], dtype="float32")
                wav_path = Path(s["path"])
            else:
                audio, sr = tts.generate_audio(state, text).numpy(), tts.sample_rate
                wav_path = Path(out_dir) / name / f"{s['source']}_{i:03d}.wav"
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                scipy.io.wavfile.write(wav_path, sr, audio)
            hyp = asr(audio, sr, lang)
            refs.setdefault(s["source"], []).append(clean(text, lang))
            hyps.setdefault(s["source"], []).append(clean(hyp, lang) or "-")
            rows.append({"source": s["source"], "ref": text, "hyp": hyp, "wav": str(wav_path)})
        cer = {src: round(jiwer.cer(refs[src], hyps[src]), 4) for src in refs}
        results[name] = {"spec": target, "cer": cer, "rows": rows}
        print(name, cer, flush=True)
        del tts
    out = Path(out_dir) / "results.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
