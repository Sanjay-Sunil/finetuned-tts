"""Build and publish the static Hugging Face Space (no server needed).

Free Spaces only serve static files, so the page plays pre-generated before/after
audio for the built-in samples (space_static/audio/{before,after}_NN.mp3) and
draws the decision slides live with rough.js. Live synthesis of any text is the
Streamlit app (app.py), run locally.

    python -m hinglish_tts.build_static_space --space zxnjxd/hinglish-pocket-tts [--upload]
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi

from hinglish_tts.decision_pages import PAGES, page_html
from hinglish_tts.samples import SAMPLES

OUT = Path("space_static")

CARD = """---
title: Hinglish Pocket TTS
emoji: 🗣️
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Before/after demo of Pocket TTS fine-tuned for Hinglish
---
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hinglish Pocket TTS</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Hind:wght@400;600&display=swap">
<style>
:root { --bg:#f8f9fa; --card:#fff; --ink:#1e1e1e; --muted:#5c636a; --line:#dee2e6;
        --before:#e8590c; --after:#1971c2; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Hind, system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 48px; }
h1 { margin:0 0 4px; font-size:2rem; }
.sub { color:var(--muted); margin:0 0 20px; }
.tabs { display:flex; gap:4px; border-bottom:1px solid var(--line); margin-bottom:20px; }
.tabs button { border:0; background:none; padding:10px 16px; font:inherit; font-size:1.05rem;
               color:var(--muted); cursor:pointer; border-bottom:3px solid transparent; }
.tabs button.on { color:var(--ink); border-bottom-color:var(--after); font-weight:600; }
.panel { display:none; } .panel.on { display:block; }
label { font-weight:600; display:block; margin-bottom:6px; }
.row { display:flex; gap:12px; align-items:stretch; }
textarea { flex:1; font:inherit; font-size:1.25rem; padding:12px; border:1px solid var(--line);
           border-radius:8px; min-height:84px; resize:vertical; background:var(--card); }
.btn { font:inherit; font-weight:600; border-radius:8px; padding:10px 18px; cursor:pointer;
       border:1px solid var(--line); background:var(--card); color:var(--ink); }
.btn.primary { color:#fff; border-color:transparent; }
.note { color:var(--muted); font-size:.95rem; margin:8px 0 20px; min-height:1.4em; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; }
.card h2 { margin:0; font-size:1.4rem; } .card p { margin:4px 0 14px; color:var(--muted); }
.card.before h2 { color:var(--before); } .card.after h2 { color:var(--after); }
.card.before .primary { background:var(--before); } .card.after .primary { background:var(--after); }
.card audio { width:100%; margin-top:12px; }
.slides iframe { width:100%; aspect-ratio:16/9; border:1px solid var(--line); border-radius:12px;
                 background:#fff; margin-bottom:16px; }
footer { margin-top:28px; color:var(--muted); font-size:.95rem; }
a { color:var(--after); }
@media (max-width:700px) { .cols { grid-template-columns:1fr; } .row { flex-direction:column; } }
</style></head><body><main>
<h1>Hinglish Pocket TTS</h1>
<p class="sub">Kyutai Pocket TTS (24-layer) fine-tuned for Hindi + English code-switching.
Both models clone the same Indian voice prompt.</p>
<div class="tabs">
  <button class="on" data-tab="try">Try it</button>
  <button data-tab="how">How it was built</button>
</div>

<section id="try" class="panel on">
  <label for="text">Text (Hindi in Devanagari, English in Latin script)</label>
  <div class="row">
    <textarea id="text"></textarea>
    <button class="btn" id="random">Random sample</button>
  </div>
  <div class="note" id="note"></div>
  <div class="cols">
    <div class="card before"><h2>Before</h2><p>released Pocket TTS (English)</p>
      <button class="btn primary" data-side="before">Play</button><audio id="a-before" controls hidden></audio></div>
    <div class="card after"><h2>After</h2><p>Hinglish fine-tune, step 4000</p>
      <button class="btn primary" data-side="after">Play</button><audio id="a-after" controls hidden></audio></div>
  </div>
</section>

<section id="how" class="panel slides">
SLIDES
</section>

<footer>Code and training details: <a href="https://github.com/Sanjay-Sunil/finetuned-tts">github.com/Sanjay-Sunil/finetuned-tts</a>.
This page plays pre-generated audio for the built-in samples; to synthesize your own text, run
<code>streamlit run app.py</code> from the repository.</footer>
</main>
<script>
const SAMPLES = SAMPLES_JSON;
const text = document.getElementById('text'), note = document.getElementById('note');
let current = 0;
function setSample(i) { current = i; text.value = SAMPLES[i]; resetAudio(); note.textContent = ''; }
function resetAudio() { for (const s of ['before', 'after']) { const a = document.getElementById('a-' + s); a.pause(); a.hidden = true; } }
document.getElementById('random').onclick = () => {
  let i; do { i = Math.floor(Math.random() * SAMPLES.length); } while (i === current && SAMPLES.length > 1);
  setSample(i);
};
text.oninput = () => { resetAudio(); note.textContent = ''; };
document.querySelectorAll('[data-side]').forEach(b => b.onclick = () => {
  const i = SAMPLES.indexOf(text.value.trim());
  if (i < 0) {
    note.textContent = 'This static page can only play the built-in samples. Pick one with "Random sample", or run the app locally for your own text.';
    return;
  }
  const side = b.dataset.side, a = document.getElementById('a-' + side);
  a.src = `audio/${side}_${String(i + 1).padStart(2, '0')}.mp3`; a.hidden = false; a.play();
});
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button, .panel').forEach(e => e.classList.remove('on'));
  b.classList.add('on'); document.getElementById(b.dataset.tab).classList.add('on');
});
setSample(0);
</script></body></html>"""

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    space: Annotated[str, typer.Option()] = "zxnjxd/hinglish-pocket-tts",
    upload: Annotated[bool, typer.Option()] = False,
):
    for i in range(1, len(SAMPLES) + 1):
        for side in ("before", "after"):
            assert (OUT / "audio" / f"{side}_{i:02d}.mp3").exists(), f"missing {side}_{i:02d}.mp3"
    (OUT / "slides").mkdir(parents=True, exist_ok=True)
    frames = []
    for i, page in enumerate(PAGES, 1):
        (OUT / "slides" / f"page_{i}.html").write_text(page_html(page))
        frames.append(f'<iframe src="slides/page_{i}.html" title="slide {i}" loading="lazy"></iframe>')
    index = PAGE.replace("SAMPLES_JSON", json.dumps(SAMPLES, ensure_ascii=False)).replace("SLIDES", "\n".join(frames))
    (OUT / "index.html").write_text(index)
    (OUT / "README.md").write_text(CARD)
    print(f"built {OUT}/index.html ({len(SAMPLES)} samples, {len(PAGES)} slides)")
    if upload:
        api = HfApi()
        api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=str(OUT), repo_id=space, repo_type="space",
                          allow_patterns=["index.html", "README.md", "audio/*.mp3", "slides/*.html"],
                          commit_message="Static demo")
        print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    app()
