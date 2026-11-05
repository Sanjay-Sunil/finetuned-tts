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

### Decisions taken

Every decision in the order it was made (times IST).

| # | When | Decision | Options considered | Choice | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | Nov 4, ~19:45 | Input script | Roman Hindi only; Devanagari + Latin; both scripts | Devanagari for Hindi, Latin for English | The human transcripts are in Devanagari, so no lossy transliteration step is needed and Hindi pronunciation stays unambiguous. The assignment allows either script. Cost: Romanized Hindi input is out of scope. |
| 2 | Nov 4, ~19:45 | Model size | 24-layer teacher only; teacher + distillation to the 6-layer CPU student | 24-layer teacher, Kyutai's `finetune_language` recipe | The budget covers one training run. Distillation needs a second run and is listed as future work. |
| 3 | Nov 4, ~19:45 | How to pick the final checkpoint | Lowest CER automatically; CER shortlist + listening | CER shortlist of two, then a listening check | CER measures intelligibility, not accent; only a listener can judge the Indian accent. |
| 4 | Nov 4, ~19:50 | First data source | IndicTTS Hindi; IndicVoices-R; LIMMITS/SYSPIN; Hinglish sets on HF | IndicTTS Hindi (SPRINGLab mirror) | Studio speech with transcripts written by people, downloadable without a form. Many HF "Hinglish" sets were rejected because their transcripts came from ASR (for example Sarvam or Whisper), which the assignment forbids. |
| 5 | Nov 4, ~20:05 | Text normalization | Leave text as is; spell out numbers and symbols | Spell out digits, `₹ % & +` (Hindi words in Devanagari sentences, Indian-English words otherwise, lakh/crore aware); danda `।` → `.` | The model can only say what it has seen written as words, and Pocket TTS splits long text only at `. ! ?`. The same function runs at inference so training and inference text match. |
| 6 | Nov 4, ~20:45 | Hindi forced aligner | `ai4bharat/indicwav2vec-hindi`; `theainerd/Wav2Vec2-large-xlsr-hindi` | `theainerd/Wav2Vec2-large-xlsr-hindi` | The AI4Bharat model is gated; the alternative is ungated and covers almost all Devanagari characters. Alignment only places the human words in time and never writes text, so it stays within the data rules. |
| 7 | Nov 4, ~20:50 | More Hindi data | IndicTTS only; all of Rasa Hindi (17.6 GB); a Rasa subset | Rasa Hindi subset: 7 shards, about 7.6 h after filtering | Rasa's command, shopping and banking styles contain English loanwords spoken by Indian speakers (*पर्सनल लोन*, *बैलेंस चेक*), the closest thing to real Hinglish in human-verified data. The full set would have taken about 2.5 h to download on the available connection. |
| 8 | Nov 4, ~20:50 | Which Rasa speech to keep | All styles; neutral styles only | Drop ANGER, DISGUST, FEAR, HAPPY, SAD, SURPRISE; keep commands, conversation, reading | Emotional acting would pull a small model away from the neutral voice wanted for a TTS demo. |
| 9 | Nov 4, ~20:55 | Training memory footprint | Smaller model; plain training at batch 64; memory-saving wrapper | `hinglish_tts/train_lowmem.py`: 8-bit AdamW, EMA weights in CPU RAM, micro-batch 4 × 16 accumulation, clips capped at 12 s | Keeps the 24-layer model and Kyutai's effective batch of 64 while cutting GPU memory (peak 5.6 GB). Upstream code stays unmodified. |
| 10 | Nov 4, ~21:10 | Fixing the first out-of-memory crash | Smaller batch; precompute codec latents separately | Run `precompute_latents --batch-size 4` before training | The crash happened in the one-time audio encoding while the full model was already on the GPU, not in training itself. |
| 11 | Nov 4, ~21:15 | English in Latin script | Wait for SPICOR Indian English (IISc); add LibriTTS-R; Devanagari-only input | One LibriTTS-R shard, 2.5 h | All Hindi data was Devanagari, so the tokenizer would never have seen Latin letters. SPICOR needs form approval that would not arrive in time. LibriTTS-R is human-transcribed and CC BY 4.0. Disclosed: its audio passed through a speech-restoration model, and its speakers are American/British. |
| 12 | Nov 4, ~21:20 | Tokenizer | Keep the English tokenizer; new SentencePiece | New unigram SentencePiece, 4,000 pieces, character coverage 1.0, fitted on all transcripts | 4,000 matches the model's text table, so no architecture change. Coverage 1.0 means no Devanagari character is ever unknown. |
| 13 | Nov 4, ~21:20 | Training length (first run) | 2,000; 3,000; 8,000 steps | 3,000 steps, checkpoint every 500 | About 2 h of training. It left time to evaluate before deciding whether to train longer. |
| 14 | Nov 4, ~21:30 | Evaluation design | WER with one ASR; per-language CER; listening only | Character error rate with a Hindi and an English wav2vec2 ASR, 10 held-out sentences per source, plus the ASR's own error on real recordings as the floor | A one-script ASR cannot fairly transcribe mixed sentences, so Hindi and English are scored separately and code-mixed sentences are judged by ear. The floor (0.16–0.18 on Hindi) shows what "perfect" looks like with this ASR. |
| 15 | Nov 4, ~21:30 | Voice prompt | Built-in English voices; a speaker from the data | One 6.7 s IndicTTS female clip, fixed for every comparison | A real Indian voice, and the same prompt everywhere keeps before/after comparisons fair. |
| 16 | Nov 4, ~22:50 | Train longer? | Stop at 3,000; extend to 6,000 | Extend to 6,000 with an unattended script that scores every 1,000-step checkpoint | Hindi CER was still falling at 1,500 steps (0.36) and English had barely moved. The script ran without supervision overnight, with sleep blocked. |
| 17 | Nov 4, ~22:50 | Raw or averaged (EMA) weights | Raw; EMA | Score both; EMA won from step 3,000 on | EMA with decay 0.999 averages over about 1,000 steps, so early on it still carries the random starting text embedding; later it smooths out the noisy raw weights. |
| 18 | Nov 5, ~01:50 | Final checkpoint | Steps 2,000–6,000, raw or EMA | Step 4,000 EMA | Best held-out CER (Hindi 0.294 / 0.341, English 0.564) and preferred over step 5,000 in the listening check. Scores worsen after 4,000–5,000 steps: the model starts overfitting, having seen the 12 h about 40 times. |
| 19 | Nov 5, ~02:00 | How to release | Raw checkpoint; official Pocket TTS layout | `hinglish_tts/package_hf.py`: weights, tokenizer, config and model card in the official layout, private repo | The stock `pocket-tts` command then runs it directly from `hf://`. The config maps `।` and curly quotes, because the stock CLI skips our normalizer. Verified from an empty cache with only a token. |
| 20 | Nov 5, ~02:00 | Dataset release | Link the sources only; upload the combined set | Private dataset repo with audio, aligned manifests and a dataset card | We combined and filtered three sources, which the assignment says requires uploading the final training set. |
| 21 | Nov 5, ~09:55 | Demo format | Screen recording; rendered video | Rendered with ffmpeg (`make_video.py`): before/after with on-screen text, two unseen sentences, then hand-drawn decision slides | Reproducible, and every clip uses the same voice prompt. The Hind font is used because Noto Sans Devanagari has no Latin letters. |

### Compute

`hinglish_tts/train_lowmem.py` wraps the upstream trainer with 8-bit AdamW and a CPU-side EMA, and `configs/hinglish_24l_local.yaml` uses micro-batches of 4 with 16-step accumulation (effective batch 64) and clips capped at 12 s, for a peak of 5.6 GB of GPU memory. The Mimi latents are precomputed in a separate process first; doing it inside the trainer ran out of memory. `modal_app.py` and `configs/hinglish_24l.yaml` hold the same pipeline for a single large GPU (batch 64, no accumulation).

### What happened during training

- The new text embedding starts random, so at step 500 the model produced Hindi-sounding speech that ignored the text (CER about 0.64, close to the English base model's 0.67).
- Hindi started following the text between steps 1,000 and 1,500; English, about 20% of the data, improved more slowly.
- With EMA decay 0.999 the averaged weights lag the raw ones early on (worse at 2,000 steps), then overtake them from 3,000 steps.


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

`modal_app.py` packages the same pipeline as Modal functions, with `configs/hinglish_24l.yaml`.

## Repository layout

| Path | What it is |
| --- | --- |
| `hinglish_tts/text_norm.py` | Number, symbol and danda normalization shared by training and inference |
| `hinglish_tts/prepare_data.py` | HF parquet datasets to 24 kHz FLAC + manifests |
| `hinglish_tts/train_lowmem.py` | Runs upstream training with less GPU memory (8-bit AdamW, CPU-side EMA) |
| `hinglish_tts/infer.py` | Inference with text normalization |
| `hinglish_tts/eval_cer.py` | CER on held-out sentences with per-language wav2vec2 ASR |
| `hinglish_tts/package_hf.py` | Export a checkpoint in the official Pocket TTS layout and upload |
| `configs/` | Training configs (gradient-accumulation and single-GPU batch 64) and the inference config |
| `modal_app.py` | Same pipeline on Modal |
| `samples/` | Baseline sentences and before/after audio |
