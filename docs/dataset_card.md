---
license: other
license_name: mixed-see-card
language:
  - hi
  - en
task_categories:
  - text-to-speech
pretty_name: Hinglish Pocket TTS training set
---

# Hinglish Pocket TTS training set

The exact training and validation data used to fine-tune `zxnjxd/pocket-tts-finetuned`: about 12.3 h of human-transcribed studio speech, 24 kHz mono FLAC, with word-level alignments.

## Sources

| Source | Language | Hours | Clips (train / valid) | Speakers | Licence |
| --- | --- | --- | --- | --- | --- |
| [SPRINGLab/IndicTTS-Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi), shard 1 of 10 | Hindi | 2.2 | 1,094 / 30 | 1 female | License for Use of Indic TTS (IIT Madras), research use |
| [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), Hindi train shards 0, 1, 11, 12, 14, 15, 16 | Hindi (with English loanwords in Devanagari) | 7.6 | 5,723 / 30 | 1 female, 1 male | CC BY 4.0 |
| [mythicinfinity/libritts_r](https://huggingface.co/datasets/mythicinfinity/libritts_r), train.clean.100 shard 0 | English | 2.5 | 1,657 / 30 | 15 | CC BY 4.0 |

All transcripts were written or verified by people. No ASR transcripts, synthetic audio or LLM-written text. LibriTTS-R audio was processed by a speech-restoration model by its authors; the speech is human.

## Preprocessing

1. Decoded, mixed to mono, resampled to 24 kHz, peak-normalized, saved as FLAC.
2. Kept clips of 1–15 s; dropped silent clips.
3. Rasa: removed the emotional styles ANGER, DISGUST, FEAR, HAPPY, SAD, SURPRISE.
4. Text normalization: digits and `₹ % & +` spelled out, danda `।` → `.`, curly quotes → straight.
5. Word timestamps by CTC forced alignment of the human transcripts (`theainerd/Wav2Vec2-large-xlsr-hindi` for Hindi, `facebook/wav2vec2-base-960h` for English). Alignment only places existing words in time.

## Layout

- `audio/<source>/*.flac`
- `train.jsonl`, `valid.jsonl`: one row per clip: `path` (relative), `duration`, `transcript`, `speaker`, `source`, `words` (`word`, `start`, `end` in seconds).
