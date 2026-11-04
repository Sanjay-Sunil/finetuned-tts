"""Generate Hinglish speech with a Pocket TTS model.

Text is normalized exactly as in training (numbers and symbols spelled out,
danda -> period) before it reaches the tokenizer.

Examples:
    # fine-tuned model from the Hugging Face repo
    python -m hinglish_tts.infer --config hf://<user>/<repo>/config.yaml \
        --voice voices/indictts_hi_female.wav --text "कल मेरा exam है, please disturb मत करना।"

    # a file of sentences, one per line -> out_dir/01.wav, 02.wav, ...
    python -m hinglish_tts.infer --config <config.yaml> --voice <wav> \
        --sentences samples/baseline_sentences.txt --out-dir outputs/finetuned
"""

from pathlib import Path
from typing import Annotated

import scipy.io.wavfile
import typer
from pocket_tts import TTSModel

from hinglish_tts.text_norm import normalize

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    voice: Annotated[str, typer.Option(help="voice prompt wav (a few seconds of speech)")],
    config: Annotated[str | None, typer.Option(help="model config yaml, local or hf://")] = None,
    language: Annotated[str | None, typer.Option(help="a released model instead of --config")] = None,
    checkpoint: Annotated[str | None, typer.Option(help="training checkpoint to load")] = None,
    text: Annotated[str | None, typer.Option()] = None,
    sentences: Annotated[str | None, typer.Option(help="text file, one sentence per line")] = None,
    out: Annotated[str, typer.Option(help="output wav for --text")] = "output.wav",
    out_dir: Annotated[str, typer.Option(help="output folder for --sentences")] = "outputs",
    prefix: Annotated[str, typer.Option(help="file name prefix for --sentences")] = "",
    temperature: Annotated[float | None, typer.Option()] = None,
    raw: Annotated[bool, typer.Option(help="skip text normalization")] = False,
):
    model = TTSModel.load_model(
        language=language, config=config, checkpoint=checkpoint, temp=temperature
    )
    state = model.get_state_for_audio_prompt(voice)

    def speak(line: str, path: Path):
        line = line if raw else normalize(line)
        audio = model.generate_audio(state, line)
        path.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(path, model.sample_rate, audio.numpy())
        print(f"{path}: {line}")

    if text:
        speak(text, Path(out))
    if sentences:
        lines = [s.strip() for s in Path(sentences).read_text().splitlines() if s.strip()]
        for i, line in enumerate(lines, 1):
            speak(line, Path(out_dir) / f"{prefix}{i:02d}.wav")


if __name__ == "__main__":
    app()
