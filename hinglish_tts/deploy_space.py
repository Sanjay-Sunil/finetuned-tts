"""Publish the Streamlit demo (app.py) as a Docker Space on Hugging Face.

The Space reads the private model with the HF_TOKEN secret, so the weights stay
private while the demo is public.

    python -m hinglish_tts.deploy_space --space zxnjxd/hinglish-pocket-tts [--set-token]
"""

from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import CommitOperationAdd, HfApi, get_token

FILES = {
    "space/README.md": "README.md",
    "space/Dockerfile": "Dockerfile",
    "space/requirements.txt": "requirements.txt",
    "app.py": "app.py",
    "hinglish_tts/__init__.py": "hinglish_tts/__init__.py",
    "hinglish_tts/text_norm.py": "hinglish_tts/text_norm.py",
    "hinglish_tts/decision_pages.py": "hinglish_tts/decision_pages.py",
    "voices/indictts_hi_female.wav": "voices/indictts_hi_female.wav",
}

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    space: Annotated[str, typer.Option()],
    private: Annotated[bool, typer.Option()] = False,
    set_token: Annotated[bool, typer.Option(help="store the local HF token as the Space's HF_TOKEN secret")] = False,
):
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    if set_token:
        api.add_space_secret(space, "HF_TOKEN", get_token())
    for src, dst in FILES.items():
        assert Path(src).exists(), src
    api.create_commit(
        repo_id=space,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src)
            for src, dst in FILES.items()
        ],
        commit_message="Deploy demo",
    )
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    app()
