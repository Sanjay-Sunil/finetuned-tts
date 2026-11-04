"""Export a training checkpoint as a Pocket TTS model folder and upload it to HF.

The folder follows the layout of the official releases, so the stock CLI can
load it straight from the Hub:

    pocket-tts generate --config hf://<repo>/config.yaml --voice <wav> --text "..."

Contents: model.safetensors (flow LM + Mimi), tokenizer.model, config.yaml,
README.md (model card), the voice prompt and the before/after samples.

    python -m hinglish_tts.package_hf runs/hinglish_24l_local/checkpoint_00003000.pt \
        --repo <user>/hinglish-pocket-tts --out dist/model [--raw] [--upload]
"""

import shutil
from pathlib import Path
from typing import Annotated

import safetensors.torch
import typer
import yaml
from huggingface_hub import HfApi
from pocket_tts import TTSModel

INFERENCE_CONFIG = "configs/inference_local.yaml"

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    checkpoint: Annotated[str, typer.Argument()],
    repo: Annotated[str, typer.Option(help="HF model repo id")],
    out: Annotated[str, typer.Option()] = "dist/model",
    raw: Annotated[bool, typer.Option(help="export raw weights instead of the EMA")] = False,
    card: Annotated[str, typer.Option(help="model card markdown")] = "docs/model_card.md",
    upload: Annotated[bool, typer.Option(help="create the private repo and push")] = False,
):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model(config=INFERENCE_CONFIG)
    model.load_training_checkpoint(checkpoint, use_ema=not raw)
    state = {f"flow_lm.{k}": v.detach().float().cpu().contiguous() for k, v in model.flow_lm.state_dict().items()}
    state.update({f"mimi.{k}": v.detach().float().cpu().contiguous() for k, v in model.mimi.state_dict().items()})
    safetensors.torch.save_file(state, str(out_dir / "model.safetensors"))

    cfg = yaml.safe_load(open(INFERENCE_CONFIG))
    shutil.copy(cfg["flow_lm"]["lookup_table"]["tokenizer_path"], out_dir / "tokenizer.model")
    cfg["weights_path"] = f"hf://{repo}/model.safetensors"
    cfg.pop("weights_path_without_voice_cloning", None)
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = f"hf://{repo}/tokenizer.model"
    # The stock CLI skips our normalizer, so map the characters it would rewrite.
    cfg["replace_characters"] = {"।": ".", "॥": ".", "’": "'", "‘": "'", "“": '"', "”": '"'}
    with open(out_dir / "config.yaml", "w") as f:
        f.write("# Hinglish Pocket TTS (24-layer), fine-tuned from Kyutai's English teacher.\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    shutil.copy(card, out_dir / "README.md")
    shutil.copytree("voices", out_dir / "voices", dirs_exist_ok=True)
    shutil.copytree("samples", out_dir / "samples", dirs_exist_ok=True)
    print(f"exported {checkpoint} ({'raw' if raw else 'EMA'}) to {out_dir}")

    if upload:
        api = HfApi()
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=repo, repo_type="model")
        print(f"uploaded to https://huggingface.co/{repo}")


if __name__ == "__main__":
    app()
