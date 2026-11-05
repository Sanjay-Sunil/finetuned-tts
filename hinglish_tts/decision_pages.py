"""Render the project's key decisions as hand-drawn (Excalidraw-style) slides.

Shapes are drawn with rough.js, text in the Kalam handwriting font (Devanagari
+ Latin), and each 1280x720 page is screenshotted with headless Chromium.

    python -m hinglish_tts.decision_pages --out-dir outputs/decisions
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

ROUGH_JS = "https://cdn.jsdelivr.net/npm/roughjs@4.6.6/bundled/rough.js"
W, H = 1280, 720

BLUE, GREEN, YELLOW, RED, GREY, VIOLET = "#a5d8ff", "#b2f2bb", "#ffec99", "#ffc9c9", "#e9ecef", "#d0bfff"

# Each element: ["box", x, y, w, h, fill, [lines], font_size]
#               ["text", x, y, size, text, color, anchor]
#               ["arrow", x1, y1, x2, y2]
#               ["line", x1, y1, x2, y2, color, dashed]
#               ["cross", x, y, size]
#               ["ellipse", cx, cy, w, h, color]
#               ["poly", [[x, y], ...], color]
PAGES = [
    [
        ["text", 640, 90, 52, "How it was built: 5 decisions", "#1e1e1e", "middle"],
        ["box", 50, 230, 200, 120, BLUE, ["1. Data"], 30],
        ["box", 295, 230, 200, 120, YELLOW, ["2. Script &", "tokenizer"], 30],
        ["box", 540, 230, 200, 120, VIOLET, ["3. Model"], 30],
        ["box", 785, 230, 200, 120, RED, ["4. Evaluation"], 30],
        ["box", 1030, 230, 200, 120, GREEN, ["5. Checkpoint"], 30],
        ["arrow", 252, 290, 293, 290], ["arrow", 497, 290, 538, 290],
        ["arrow", 742, 290, 783, 290], ["arrow", 987, 290, 1028, 290],
        ["text", 150, 410, 22, "12.3 h of human-", "#1e1e1e", "middle"],
        ["text", 150, 440, 22, "verified speech", "#1e1e1e", "middle"],
        ["text", 395, 410, 22, "Devanagari + Latin,", "#1e1e1e", "middle"],
        ["text", 395, 440, 22, "new 4k SentencePiece", "#1e1e1e", "middle"],
        ["text", 640, 410, 22, "24-layer teacher,", "#1e1e1e", "middle"],
        ["text", 640, 440, 22, "new-language recipe", "#1e1e1e", "middle"],
        ["text", 885, 410, 22, "per-language CER", "#1e1e1e", "middle"],
        ["text", 885, 440, 22, "+ ASR floor + listening", "#1e1e1e", "middle"],
        ["text", 1130, 410, 22, "step 4000 (EMA)", "#1e1e1e", "middle"],
        ["text", 1130, 440, 22, "by CER + by ear", "#1e1e1e", "middle"],
        ["text", 640, 600, 26, "every choice + its reason: README, Decisions taken", "#495057", "middle"],
    ],
    [
        ["text", 640, 75, 46, "1 · Data: human-verified only", "#1e1e1e", "middle"],
        ["box", 60, 130, 470, 105, BLUE, ["IndicTTS Hindi (IIT Madras) · 2.2 h", "studio, literary Hindi"], 24],
        ["box", 60, 260, 470, 105, BLUE, ["Rasa Hindi (AI4Bharat) · 7.6 h", "commands: पर्सनल लोन, बैलेंस चेक"], 24],
        ["box", 60, 390, 470, 105, BLUE, ["LibriTTS-R English · 2.5 h", "teaches Latin-script English"], 24],
        ["arrow", 535, 183, 760, 300], ["arrow", 535, 313, 760, 313], ["arrow", 535, 443, 760, 326],
        ["box", 770, 240, 440, 150, GREEN, ["12.3 h training set", "all transcripts written", "by people"], 28],
        ["text", 60, 560, 26, "Rejected:", "#c92a2a", "start"],
        ["cross", 75, 600, 16], ["text", 100, 608, 22, "ASR/Whisper-labelled \"Hinglish\" sets (breaks the rule)", "#1e1e1e", "start"],
        ["cross", 75, 645, 16], ["text", 100, 653, 22, "full Rasa, 17.6 GB (too slow to download)", "#1e1e1e", "start"],
        ["cross", 75, 690, 16], ["text", 100, 698, 22, "SPICOR Indian English (form approval too slow)", "#1e1e1e", "start"],
    ],
    [
        ["text", 640, 75, 46, "2 + 3 · What changed inside the model", "#1e1e1e", "middle"],
        ["text", 120, 205, 26, "कल मेरा exam है", "#1e1e1e", "middle"],
        ["arrow", 120, 225, 120, 275],
        ["box", 20, 280, 200, 120, YELLOW, ["tokenizer +", "text embedding"], 24],
        ["text", 120, 430, 24, "NEW", "#e67700", "middle"],
        ["arrow", 222, 340, 288, 340],
        ["box", 290, 280, 230, 120, BLUE, ["24-layer", "backbone"], 26],
        ["text", 405, 430, 24, "fine-tuned", "#1971c2", "middle"],
        ["arrow", 522, 340, 588, 340],
        ["box", 590, 280, 200, 120, BLUE, ["LSD head", "(1 step/frame)"], 24],
        ["text", 690, 430, 24, "fine-tuned", "#1971c2", "middle"],
        ["arrow", 792, 340, 858, 340],
        ["box", 860, 280, 200, 120, GREY, ["Mimi codec", "decoder"], 24],
        ["text", 960, 430, 24, "frozen", "#495057", "middle"],
        ["arrow", 1062, 340, 1128, 340],
        ["text", 1190, 350, 30, "audio", "#1e1e1e", "middle"],
        ["box", 290, 500, 230, 80, GREY, ["voice prompt", "(5 s, Indian female)"], 22],
        ["arrow", 405, 498, 405, 402],
        ["text", 640, 640, 24, "English tokenizer has no Devanagari → new tokenizer, fresh text embedding", "#495057", "middle"],
        ["text", 640, 680, 24, "Script: Devanagari for Hindi, Latin for English (matches the human transcripts)", "#495057", "middle"],
    ],
    [
        ["text", 640, 75, 46, "4 · How we measured", "#1e1e1e", "middle"],
        ["box", 60, 140, 340, 120, BLUE, ["Hindi sentences", "\u2192 Hindi wav2vec2 ASR"], 26],
        ["box", 470, 140, 340, 120, BLUE, ["English sentences", "\u2192 English wav2vec2 ASR"], 26],
        ["box", 880, 140, 340, 120, YELLOW, ["mixed sentences", "\u2192 listen"], 26],
        ["arrow", 230, 262, 230, 330], ["arrow", 640, 262, 640, 330], ["arrow", 1050, 262, 1050, 330],
        ["box", 60, 335, 750, 100, GREEN, ["character error rate on 10 held-out sentences per source"], 26],
        ["box", 880, 335, 340, 100, GREEN, ["accent, smoothness", "before vs after"], 24],
        ["text", 60, 510, 26, "\u2022 floor: the ASR on real recordings scores 0.16\u20130.18 (Hindi), so ~0.17 = perfect", "#1e1e1e", "start"],
        ["text", 60, 560, 26, "\u2022 one-script ASR cannot score mixed sentences fairly \u2192 those are judged by ear", "#1e1e1e", "start"],
        ["text", 60, 610, 26, "\u2022 raw and averaged (EMA) weights scored at every 1,000 steps", "#1e1e1e", "start"],
        ["text", 60, 660, 26, "\u2022 same Indian voice prompt for every model and sentence", "#1e1e1e", "start"],
    ],
    [
        ["text", 640, 70, 46, "5 · Which checkpoint?", "#1e1e1e", "middle"],
        ["line", 120, 600, 1000, 600, "#1e1e1e", False], ["line", 120, 600, 120, 130, "#1e1e1e", False],
        ["text", 560, 650, 22, "training step", "#495057", "middle"],
        ["text", 130, 112, 22, "CER", "#495057", "start"],
        ["text", 105, 600 - 0.2 * 600 + 7, 18, "0.2", "#495057", "end"],
        ["text", 105, 600 - 0.4 * 600 + 7, 18, "0.4", "#495057", "end"],
        ["text", 105, 600 - 0.6 * 600 + 7, 18, "0.6", "#495057", "end"],
        ["text", 105, 600 - 0.8 * 600 + 7, 18, "0.8", "#495057", "end"],
        ["line", 120, 600 - 0.18 * 600, 1000, 600 - 0.18 * 600, "#2f9e44", True],
        ["text", 1010, 600 - 0.18 * 600 + 6, 20, "real recordings, Hindi (floor)", "#2f9e44", "start"],
        ["line", 120, 600 - 0.67 * 600, 1000, 600 - 0.67 * 600, "#868e96", True],
        ["text", 1010, 600 - 0.67 * 600 + 6, 20, "released model, on Hindi", "#868e96", "start"],
        ["poly", [[247, 600 - 0.48 * 600], [393, 600 - 0.37 * 600], [540, 600 - 0.35 * 600],
                  [687, 600 - 0.29 * 600], [833, 600 - 0.29 * 600], [980, 600 - 0.34 * 600]], "#1971c2"],
        ["text", 1010, 600 - 0.34 * 600 + 6, 22, "Hindi (ours)", "#1971c2", "start"],
        ["poly", [[247, 600 - 0.73 * 600], [393, 600 - 0.66 * 600], [540, 600 - 0.64 * 600],
                  [687, 600 - 0.56 * 600], [833, 600 - 0.68 * 600], [980, 600 - 0.64 * 600]], "#e8590c"],
        ["text", 247, 600 - 0.73 * 600 - 22, 22, "English (ours)", "#e8590c", "middle"],
        ["ellipse", 687, 600 - 0.43 * 600, 90, 260, "#c92a2a"],
        ["text", 687, 140, 24, "picked: step 4000", "#c92a2a", "middle"],
        ["text", 247, 625, 20, "1k", "#495057", "middle"], ["text", 393, 625, 20, "2k", "#495057", "middle"],
        ["text", 540, 625, 20, "3k", "#495057", "middle"], ["text", 687, 625, 20, "4k", "#495057", "middle"],
        ["text", 833, 625, 20, "5k", "#495057", "middle"], ["text", 980, 625, 20, "6k", "#495057", "middle"],
        ["text", 640, 700, 22, "best held-out CER and preferred by ear; after 4k it overfits (12 h seen ~40×)", "#495057", "middle"],
    ],
]

HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
@font-face { font-family: Hand; src: url("file://FONT_PATH"); }
html, body { margin: 0; background: #ffffff; }
svg { display: block; }
text { font-family: Hand, sans-serif; }
</style>
<script src="ROUGH_JS"></script></head><body>
<svg id="s" width="W" height="H" viewBox="0 0 W H"></svg>
<script>
const els = ELEMENTS;
const svg = document.getElementById('s');
const rc = rough.svg(svg);
const NS = 'http://www.w3.org/2000/svg';
let seed = 7;
const opt = (o) => Object.assign({roughness: 1.4, bowing: 1.2, stroke: '#1e1e1e', strokeWidth: 2, seed: seed++}, o);
function text(x, y, size, str, color, anchor) {
  const t = document.createElementNS(NS, 'text');
  t.setAttribute('x', x); t.setAttribute('y', y); t.setAttribute('font-size', size);
  t.setAttribute('fill', color); t.setAttribute('text-anchor', anchor);
  t.textContent = str; svg.appendChild(t);
}
function arrow(x1, y1, x2, y2) {
  svg.appendChild(rc.line(x1, y1, x2, y2, opt({})));
  const a = Math.atan2(y2 - y1, x2 - x1), l = 16;
  for (const d of [0.45, -0.45]) {
    svg.appendChild(rc.line(x2, y2, x2 - l * Math.cos(a + d), y2 - l * Math.sin(a + d), opt({})));
  }
}
for (const e of els) {
  const k = e[0];
  if (k === 'box') {
    const [, x, y, w, h, fill, lines, size] = e;
    svg.appendChild(rc.rectangle(x, y, w, h, opt({fill, fillStyle: 'hachure', hachureGap: 7, fillWeight: 1.6})));
    const lh = size * 1.25, top = y + h / 2 - (lines.length - 1) * lh / 2 + size * 0.35;
    lines.forEach((ln, i) => text(x + w / 2, top + i * lh, size, ln, '#1e1e1e', 'middle'));
  } else if (k === 'text') { const [, x, y, size, s, c, a] = e; if (s) text(x, y, size, s, c, a); }
  else if (k === 'arrow') { arrow(e[1], e[2], e[3], e[4]); }
  else if (k === 'line') {
    const [, x1, y1, x2, y2, c, dashed] = e;
    const o = opt({stroke: c}); if (dashed) o.strokeLineDash = [10, 8];
    svg.appendChild(rc.line(x1, y1, x2, y2, o));
  } else if (k === 'cross') {
    const [, x, y, s] = e;
    svg.appendChild(rc.line(x - s, y - s, x + s, y + s, opt({stroke: '#c92a2a', strokeWidth: 3})));
    svg.appendChild(rc.line(x + s, y - s, x - s, y + s, opt({stroke: '#c92a2a', strokeWidth: 3})));
  } else if (k === 'ellipse') {
    const [, cx, cy, w, h, c] = e;
    svg.appendChild(rc.ellipse(cx, cy, w, h, opt({stroke: c, strokeWidth: 3})));
  } else if (k === 'poly') {
    const [, pts, c] = e;
    svg.appendChild(rc.linearPath(pts, opt({stroke: c, strokeWidth: 3.5})));
    for (const [x, y] of pts) svg.appendChild(rc.circle(x, y, 12, opt({stroke: c, fill: c, fillStyle: 'solid'})));
  }
}
</script></body></html>"""

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    out_dir: Annotated[str, typer.Option()] = "outputs/decisions",
    font: Annotated[str, typer.Option()] = str(Path.home() / ".local/share/fonts/kalam/Kalam-Regular.ttf"),
    chromium: Annotated[str, typer.Option()] = "chromium",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        for i, page in enumerate(PAGES, 1):
            html = (HTML.replace("ELEMENTS", json.dumps(page, ensure_ascii=False))
                    .replace("FONT_PATH", font).replace("ROUGH_JS", ROUGH_JS)
                    .replace('"W"', f'"{W}"').replace('"H"', f'"{H}"')
                    .replace("0 0 W H", f"0 0 {W} {H}"))
            src = Path(d) / f"page_{i}.html"
            src.write_text(html)
            png = out / f"page_{i}.png"
            subprocess.run(
                [chromium, "--headless", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                 f"--window-size={W},{H}", "--virtual-time-budget=5000", f"--screenshot={png.resolve()}",
                 src.as_uri()],
                check=True, capture_output=True,
            )
            print(png)


if __name__ == "__main__":
    app()
