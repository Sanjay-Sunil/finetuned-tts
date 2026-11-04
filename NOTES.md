# NOTES — Hinglish Pocket TTS

Running log: decisions (with my reasoning), learnings, failures. Becomes the README "Approach" section.

## Decision log
| # | Decision | Options considered | Choice | Why | Date |
|---|----------|--------------------|--------|-----|------|
| 1 | Dataset | IndicTTS Hindi only; + Indian English; + Rasa Hindi (all or subset) | IndicTTS Hindi + Rasa Hindi subset (command/conversational styles) + SPICOR English if access arrives | Studio recordings, human transcripts. HF mirror (SPRINGLab/IndicTTS-Hindi) is ungated but subject to the "License For Use of Indic TTS" (IIT Madras), not plain CC BY. SPRINGLab has no English subset → Indian English candidate: SPICOR TTS_1.0 (IISc SPIRE, human quality-checked, form access). Fits ~10–20 h budget. | 2026-11-04 |
| 2 | Tokenizer / script | Devanagari+Latin; Roman only; both | Devanagari (Hindi) + Latin (English), new SentencePiece | Matches dataset script, so no lossy transliteration step. Email allows either script. Cost: Roman-Hindi input is out of scope. | 2026-11-04 |
| 3 | Model size / recipe | 24l only; 24l + distill to 6l | 24l teacher, finetune_language recipe, no distillation | $30 / ~5 h budget only covers one training run; distillation listed as future work. | 2026-11-04 |
| 4 | Final checkpoint | auto by CER; CER shortlist + listening | CER shortlist of 2, user listens and picks | CER measures intelligibility, not accent; a human ear is needed for accent. | 2026-11-04 (method) |

## Hour 1 — Understanding the model


1. Why does CALM avoid RVQ tokens? Gain vs. new problem?
2. What does the backbone output each step; what does the MLP head do with it?
3. Why a consistency head instead of diffusion? Tie to CPU speed.
4. Why is temperature tricky for consistency models; what did they do?
5. What trick prevents autoregressive error accumulation?
6. Where does text enter; where does the voice prompt enter; what is cloned?
7. For Hinglish: which parts are language-specific vs. language-agnostic (codec, text embedding, backbone, head)?

## Baseline (before fine-tuning)
- Voice prompt used:
- Observations per sentence (baseline_01..08.wav):

## Data
- Source(s), license, link:
- Preprocessing:

## Training runs
| Run | Config | Steps | GPU time / $ | Notes |
|-----|--------|-------|--------------|-------|

## Evaluation (CER on held-out)
| Model | CER | Notes |
|-------|-----|-------|

## Failures & fixes

## What I'd do with more compute

### 2026-11-04 data notes
- Rasa (ai4bharat/Rasa, Hindi, CC BY 4.0) considered for more Hindi incl. voice-command speech; not accessible from our account, dropped.
- IndicTTS Hindi transcripts are literary Hindi (epics, no digits, no Latin words). Code-switching therefore relies on the pretrained English backbone plus English data.
- Modal GPU functions require a payment method; card declined. Backup plan: local RTX 4060 Laptop (8 GB, bf16-capable).
- Rasa Hindi profile: 45.8 h (24.4 F / 21.4 M), no Latin script, but command/shopping/banking styles (ALEXA, BB, DIGI, UMANG, CONV) contain many English loanwords written in Devanagari ("साउंड म्यूट करो", "पर्सनल लोन"). Chosen as a subset (~8–10 h) because local bandwidth (~2 MB/s) makes the full 17.6 GB impractical.
