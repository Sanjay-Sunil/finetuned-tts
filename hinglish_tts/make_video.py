"""Render the before/after demo video with ffmpeg.

Each segment shows the sentence on screen with a live waveform, first for the
released English Pocket TTS ("before") and then for the fine-tuned model
("after"), both cloning the same voice prompt.

    python -m hinglish_tts.make_video --out outputs/demo.mp4
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import soundfile as sf
import typer

FONT = str(Path.home() / ".local/share/fonts/hind/Hind-Regular.ttf")  # Devanagari + Latin
FONT_BOLD = str(Path.home() / ".local/share/fonts/hind/Hind-SemiBold.ttf")
W, H = 1280, 720

# (sentence, before wav, after wav)
SEGMENTS = [
    (4, "samples/baseline/baseline_04.wav", "samples/finetuned/finetuned_04.wav"),
    (5, "samples/baseline/baseline_05.wav", "samples/finetuned/finetuned_05.wav"),
    (8, "samples/baseline/baseline_08.wav", "samples/finetuned/finetuned_08.wav"),
    ("x1", "samples/baseline/extra_01.wav", "samples/finetuned/extra_01.wav"),
    ("x2", "samples/baseline/extra_02.wav", "samples/finetuned/extra_02.wav"),
]

app = typer.Typer(pretty_exceptions_show_locals=False)


def textfile(tmp: Path, name: str, text: str) -> str:
    p = tmp / f"{name}.txt"
    p.write_text(text)
    return str(p)


def card(tmp: Path, name: str, lines: list[tuple[str, int, int]], seconds: float) -> Path:
    """A static title card: lines of (text, font size, y)."""
    draws = [
        f"drawtext=fontfile={FONT_BOLD if i == 0 else FONT}:textfile={textfile(tmp, f'{name}_{i}', t)}"
        f":fontsize={size}:fontcolor=white:x=(w-text_w)/2:y={y}"
        for i, (t, size, y) in enumerate(lines)
    ]
    out = tmp / f"{name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=0x101418:s={W}x{H}:d={seconds}",
         "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono:d={seconds}",
         "-vf", ",".join(draws), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
         "-c:a", "aac", "-ar", "24000", str(out)],
        check=True,
    )
    return out


def clip(tmp: Path, name: str, sentence: str, label: str, color: str, wav: str) -> Path:
    """Sentence + label on top, waveform of `wav` below, audio playing."""
    dur = sf.info(wav).duration + 0.6
    pic = tmp / f"{name}_wave.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-lavfi",
         f"showwavespic=s={W - 160}x260:colors={color}", "-frames:v", "1", str(pic)],
        check=True,
    )
    # Static waveform with a playhead sweeping across it while the clip plays.
    span = sf.info(wav).duration
    filt = (
        f"[0:v][2:v]overlay=80:360[v0];"
        f"[3:v]format=rgba[bar];[v0][bar]overlay=x='80+{W - 160}*min(t/{span:.3f}\\,1)':y=360[v1];"
        f"[v1]drawtext=fontfile={FONT_BOLD}:textfile={textfile(tmp, name + '_l', label)}:fontsize=34"
        f":fontcolor={color}:x=80:y=70,"
        f"drawtext=fontfile={FONT}:textfile={textfile(tmp, name + '_s', sentence)}:fontsize=40"
        f":fontcolor=white:x=80:y=170:line_spacing=12[v]"
    )
    out = tmp / f"{name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=0x101418:s={W}x{H}:d={dur}",
         "-i", wav, "-loop", "1", "-i", str(pic), "-f", "lavfi", "-i", f"color=c=white:s=3x260:d={dur}",
         "-filter_complex", filt, "-map", "[v]", "-map", "1:a", "-af", "apad=pad_dur=0.6",
         "-t", f"{dur}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-ar", "24000",
         str(out)],
        check=True,
    )
    return out


def still(tmp: Path, name: str, png: Path, seconds: float) -> Path:
    """A slide image held for `seconds`, with silent audio so the concat stays in sync."""
    out = tmp / f"{name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png),
         "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono:d={seconds}",
         "-vf", f"scale={W}:{H},format=yuv420p", "-t", f"{seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-r", "25", "-c:a", "aac", "-ar", "24000", str(out)],
        check=True,
    )
    return out


def wrap(text: str, width: int = 44) -> str:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return "\n".join(lines + [cur])


@app.command()
def main(
    out: Annotated[str, typer.Option()] = "outputs/demo.mp4",
    pages: Annotated[str, typer.Option(help="folder of decision slides (page_N.png) appended at the end")] = "outputs/decisions",
    page_seconds: Annotated[float, typer.Option()] = 5.0,
):
    baseline = Path("samples/baseline_sentences.txt").read_text().splitlines()
    extra = Path("samples/demo_extra_sentences.txt").read_text().splitlines()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        parts = [card(tmp, "title", [
            ("Hinglish Pocket TTS", 64, 230),
            ("Kyutai Pocket TTS (24-layer) fine-tuned on human-verified Hindi + English speech", 28, 340),
            ("Same Indian voice prompt for every clip.  BEFORE = released English model,", 26, 420),
            ("AFTER = this fine-tune (step 4000).  Last two sentences were never seen in training.", 26, 460),
        ], 6)]
        for i, (key, before, after) in enumerate(SEGMENTS):
            text = baseline[key - 1] if isinstance(key, int) else extra[int(key[1:]) - 1]
            text = wrap(text)
            parts.append(clip(tmp, f"s{i}b", text, "BEFORE  ·  released Pocket TTS (English)", "0xff8a65", before))
            parts.append(clip(tmp, f"s{i}a", text, "AFTER  ·  Hinglish fine-tune", "0x4dd0e1", after))
        parts.append(card(tmp, "end", [
            ("Model: huggingface.co/zxnjxd/pocket-tts-finetuned (private)", 30, 300),
            ("Code: github.com/Sanjay-Sunil/finetuned-tts", 30, 360),
        ], 4))
        for i, png in enumerate(sorted(Path(pages).glob("page_*.png")), 1):
            parts.append(still(tmp, f"page{i}", png, page_seconds))
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", out], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
