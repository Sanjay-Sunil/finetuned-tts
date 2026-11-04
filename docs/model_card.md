---
license: cc-by-4.0
language:
  - hi
  - en
base_model: kyutai/pocket-tts
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - hinglish
  - code-switching
  - pocket-tts
---

# Hinglish Pocket TTS (24-layer)

Kyutai's 24-layer Pocket TTS teacher fine-tuned for Indian Hinglish: Hindi in Devanagari with English words in Latin script, in one sentence, with zero-shot voice cloning from a short prompt.

## Use

```bash
pip install pocket-tts
pocket-tts generate --config hf://zxnjxd/pocket-tts-finetuned/config.yaml \
    --voice hf://zxnjxd/pocket-tts-finetuned/voices/indictts_hi_female.wav \
    --text "कल मेरा exam है, please disturb मत करना।"
```

Write numbers and symbols as words, or use `hinglish_tts.infer` from the training repository, which normalizes them (`2,800` becomes `दो हज़ार आठ सौ`).

Input script: Devanagari for Hindi, Latin for English. Romanized Hindi ("Mujhe lagta hai") is outside the training distribution.

## Training

- Base: `kyutai/pocket-tts` `english_2026-04_24l`, text embedding re-initialized for a new 4,000-piece SentencePiece tokenizer (Devanagari + Latin).
- Objective: LSD flow head, Kyutai's `finetune_language` recipe (lr 2e-4, constant, 500 warm-up steps), effective batch 64.
- Data: about 12 h of human-transcribed speech: IndicTTS Hindi, Rasa Hindi (AI4Bharat), LibriTTS-R English. See the dataset repository for details.

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

## Limitations

- Trained on about 12 h, far below the 100+ h the Pocket TTS authors recommend; pronunciation of rare words is unreliable.
- English words in Latin script are the weakest part (see the CER table): only 2.5 h of English, read by American and British speakers, so English words lean towards those accents.
- Romanized Hindi input is out of scope.
- Very little truly code-mixed training speech; code-switching relies on Rasa's Devanagari loanwords plus the English data.
- 24-layer model only: slower on CPU than the released 6-layer Pocket TTS.
