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
pocket-tts generate --config hf://REPO_ID/config.yaml \
    --voice hf://REPO_ID/voices/indictts_hi_female.wav \
    --text "कल मेरा exam है, please disturb मत करना।"
```

Write numbers and symbols as words, or use `hinglish_tts.infer` from the training repository, which normalizes them (`2,800` becomes `दो हज़ार आठ सौ`).

Input script: Devanagari for Hindi, Latin for English. Romanized Hindi ("Mujhe lagta hai") is outside the training distribution.

## Training

- Base: `kyutai/pocket-tts` `english_2026-04_24l`, text embedding re-initialized for a new 4,000-piece SentencePiece tokenizer (Devanagari + Latin).
- Objective: LSD flow head, Kyutai's `finetune_language` recipe (lr 2e-4, constant, 500 warm-up steps), effective batch 64.
- Data: about 12 h of human-transcribed speech: IndicTTS Hindi, Rasa Hindi (AI4Bharat), LibriTTS-R English. See the dataset repository for details.

## Results

RESULTS_TABLE

## Limitations

LIMITATIONS
