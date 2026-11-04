"""Bundle the combined training set for the private HF dataset repo.

Audio is hard-linked into dist/dataset/audio/<source>/ and the aligned
manifests are rewritten with paths relative to the repo root.

    python -m hinglish_tts.package_dataset --repo zxnjxd/pocket-tts-finetuned-data [--upload]
"""

import json
import os
import shutil
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    repo: Annotated[str, typer.Option()],
    mix: Annotated[str, typer.Option()] = "data/local/mix",
    out: Annotated[str, typer.Option()] = "dist/dataset",
    card: Annotated[str, typer.Option()] = "docs/dataset_card.md",
    upload: Annotated[bool, typer.Option()] = False,
):
    out_dir = Path(out)
    for split in ("train", "valid"):
        rows = []
        for line in open(Path(mix) / f"{split}_aligned.jsonl"):
            r = json.loads(line)
            src_path = Path(r["path"])
            rel = Path("audio") / r["source"] / src_path.name
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.link(src_path, dst)
            r["path"] = str(rel)
            rows.append(r)
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} h")
    shutil.copy(card, out_dir / "README.md")
    if upload:
        api = HfApi()
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_large_folder(folder_path=str(out_dir), repo_id=repo, repo_type="dataset")
        print(f"uploaded to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    app()
