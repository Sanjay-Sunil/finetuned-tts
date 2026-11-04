# Hinglish Pocket TTS

A fine-tune of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (24-layer teacher) for Indian Hinglish: Hindi in Devanagari, English words in Latin script, in the same sentence, with an Indian accent.

- Model (private): `https://huggingface.co/zxnjxd/pocket-tts-finetuned`
- Training data (private): `https://huggingface.co/datasets/zxnjxd/pocket-tts-finetuned-data`

## Quick start

```bash
pip install pocket-tts
pocket-tts generate --config hf://zxnjxd/pocket-tts-finetuned/config.yaml \
    --voice voices/indictts_hi_female.wav \
    --text "कल मेरा exam है, please disturb मत करना।"
```

Numbers and symbols must be written as words for the stock CLI. `hinglish_tts.infer` does that for you:

```bash
python -m hinglish_tts.infer --config hf://zxnjxd/pocket-tts-finetuned/config.yaml \
    --voice voices/indictts_hi_female.wav --text "मुझे 2,800 रुपये चाहिए, please."
```

## Results

Character error rate (CER, lower is better) on 10 held-out sentences per source, voice prompt `voices/indictts_hi_female.wav`, greedy wav2vec2 CTC transcription (`hinglish_tts/eval_cer.py`). "Real recordings" is the ASR's error on the human audio itself, the best score reachable with this ASR.

| Model | IndicTTS Hindi | Rasa Hindi | English (LibriTTS-R) |
| --- | --- | --- | --- |
| Real recordings (ASR floor) | 0.179 | 0.156 | 0.010 |
| Released Pocket TTS (English) | ~0.67 | ~0.69 | 0.006 |
| Step 1,000 | 0.482 | 0.552 | 0.731 |
| Step 2,000 | 0.372 | 0.399 | 0.660 |
| Step 3,000 (EMA) | 0.350 | 0.346 | 0.636 |
| **Step 4,000 (EMA), released** | **0.294** | **0.341** | **0.564** |
| Step 5,000 (EMA) | 0.289 | 0.356 | 0.682 |
| Step 6,000 (EMA) | 0.337 | 0.386 | 0.638 |

Step 4,000 is released: best overall, preferred in a listening comparison with step 5,000; later steps overfit (about 40 passes over 12 h of data). Code-mixed sentences are not scored, because a single-language ASR cannot transcribe them fairly; they are in `samples/` as before/after audio (`baseline_0N.wav` = released English model, `finetuned_0N.wav` = this model, same voice prompt). With 10 sentences per source, differences under about 0.03 are within noise.

## Data

| Source | Language | Hours used | Speakers | Transcripts | Licence |
| --- | --- | --- | --- | --- | --- |
| [IndicTTS Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) (IIT Madras) | Hindi | 2.2 | 1 (female) | Human, studio read speech | Indic TTS licence (research use) |
| [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) Hindi (AI4Bharat) | Hindi, incl. English loanwords in Devanagari | 7.6 | 2 | Human, studio read speech | CC BY 4.0 |
| [LibriTTS-R](https://huggingface.co/datasets/mythicinfinity/libritts_r) train-clean-100, one shard | English | 2.5 | 15 | Human (book text, aligned) | CC BY 4.0 |

No ASR-generated transcripts, TTS-generated audio or LLM-written text is used. Two disclosures: LibriTTS-R audio is LibriTTS passed through a speech-restoration model (the speech itself is human); and word timestamps come from wav2vec2 forced alignment of the human transcripts, which places existing words in time and never writes text.

Preprocessing (`hinglish_tts/prepare_data.py`, `hinglish_tts/text_norm.py`):

1. Decode, mix to mono, resample to 24 kHz, peak-normalize, store as FLAC.
2. Keep clips of 1–15 s; drop silent clips. Rasa: drop the emotional styles (anger, disgust, fear, happy, sad, surprise) and keep command, shopping, banking, conversational and reading styles.
3. Normalize text: digits to words (Hindi words in Devanagari sentences, Indian-English words otherwise, lakh/crore aware), `₹ % & +` to words, danda `।` to `.`, curly quotes to straight.
4. Hold out 15–20 clips per speaker group for validation.
5. Forced alignment: `theainerd/Wav2Vec2-large-xlsr-hindi` for Hindi, `facebook/wav2vec2-base-960h` for English (upstream `training/scripts/align_data.py`).
6. Tokenizer: SentencePiece unigram, 4,000 pieces, character coverage 1.0, fitted on all transcripts.

## Approach

### Decisions

| # | Decision | Choice | Why |
| --- | --- | --- | --- |
| 1 | Data | IndicTTS Hindi + Rasa Hindi subset + one LibriTTS-R shard | Studio speech with human transcripts only. Rasa's command, shopping and banking styles contain English loanwords spoken by Indian speakers; LibriTTS-R teaches Latin-script English. Subset sized to a slow connection. |
| 2 | Script / tokenizer | Devanagari for Hindi, Latin for English; new 4,000-piece SentencePiece | Matches the human transcripts, so no lossy transliteration. The released English tokenizer has no Devanagari. |
| 3 | Model | 24-layer teacher, `finetune_language` recipe, no distillation | One run fits the budget; distilling to the 6-layer CPU student is future work. |
| 4 | Checkpoint | Lowest held-out CER, then a listening check | CER measures intelligibility, not accent. |

### Compute

Modal would not accept a payment card, so training ran on a laptop RTX 4060 (8 GB). `hinglish_tts/train_lowmem.py` wraps the upstream trainer with 8-bit AdamW and a CPU-side EMA, and the recipe uses micro-batches of 4 with 16-step accumulation (effective batch 64) and clips capped at 12 s. Peak memory 5.6 GB, about 0.45 steps/s: 6,000 steps in about 3.7 h. The Mimi latents are precomputed in a separate process first; doing it inside the trainer ran out of memory.

### What happened during training

- The new text embedding starts random, so at step 500 the model produced Hindi-sounding speech that ignored the text (CER about 0.64, close to the English base model's 0.67).
- Hindi started following the text between steps 1,000 and 1,500; English, about 20% of the data, improved more slowly.
- With EMA decay 0.999 the averaged weights lag the raw ones early on (worse at 2,000 steps), then overtake them from 3,000 steps.

Full log of measurements and failures: `NOTES.md`.

## Reproduce

```bash
git clone https://github.com/kyutai-labs/pocket-tts third_party/pocket-tts
git -C third_party/pocket-tts checkout 2e117a1e7df06323c9df0315c671f3cbc3b8d9fc
python -m venv .venv-gpu && .venv-gpu/bin/pip install -e third_party/pocket-tts \
    transformers sphn jiwer pyyaml tqdm soundfile pyarrow scipy typer sentencepiece bitsandbytes

# data (see the Data section for the exact shards)
python -m hinglish_tts.prepare_data SPRINGLab/IndicTTS-Hindi data/local/indictts_hi --source indictts_hi --speaker-col gender
python -m hinglish_tts.prepare_data ai4bharat/Rasa data/local/rasa_hi --source rasa_hi --prefix Hindi/ \
    --speaker-col gender --style-col style --exclude-styles ANGER,DISGUST,FEAR,HAPPY,SAD,SURPRISE --max-sec 15
python -m hinglish_tts.prepare_data mythicinfinity/libritts_r data/local/libritts_en --source libritts_en \
    --prefix data/train.clean.100/ --text-col text_normalized --speaker-col speaker_id --max-sec 15

# from third_party/pocket-tts: align, tokenizer, latents, train
python -m training.scripts.align_data <in.jsonl> <out_aligned.jsonl> --model <ctc model>
python -m training.scripts.train_tokenizer ../../data/local/mix/tokenizer ../../data/local/mix/*_aligned.jsonl --character-coverage 1.0
python -m training.scripts.precompute_latents ../../configs/hinglish_24l_local.yaml --batch-size 4
PYTHONPATH=../.. python -m hinglish_tts.train_lowmem ../../configs/hinglish_24l_local.yaml

# evaluate and package (from the repo root)
python -m hinglish_tts.eval_cer --voice voices/indictts_hi_female.wav --model gt=reference \
    --model base=language:english --model ft=configs/inference_local.yaml:runs/hinglish_24l_local/checkpoint_00003000.pt
python -m hinglish_tts.package_hf runs/hinglish_24l_local/checkpoint_00003000.pt --repo zxnjxd/pocket-tts-finetuned --upload
```

`modal_app.py` runs the same pipeline on a Modal H100 (batch 64, no accumulation) with `configs/hinglish_24l.yaml`.

## Repository layout

| Path | What it is |
| --- | --- |
| `hinglish_tts/text_norm.py` | Number, symbol and danda normalization shared by training and inference |
| `hinglish_tts/prepare_data.py` | HF parquet datasets to 24 kHz FLAC + manifests |
| `hinglish_tts/train_lowmem.py` | Runs upstream training on an 8 GB GPU (8-bit AdamW, CPU-side EMA) |
| `hinglish_tts/infer.py` | Inference with text normalization |
| `hinglish_tts/eval_cer.py` | CER on held-out sentences with per-language wav2vec2 ASR |
| `hinglish_tts/package_hf.py` | Export a checkpoint in the official Pocket TTS layout and upload |
| `configs/` | Training configs (laptop and Modal) and the local inference config |
| `modal_app.py` | Same pipeline on Modal |
| `samples/` | Baseline sentences and before/after audio |
| `NOTES.md` | Decision log, measurements and failures |
