# NOTES — Hinglish Pocket TTS

Running log: decisions (with my reasoning), learnings, failures. Becomes the README "Approach" section.

## Decision log
| # | Decision | Options considered | Choice | Why | Date |
|---|----------|--------------------|--------|-----|------|
| 1 | Dataset | IndicTTS Hindi only; + Indian English; + Rasa Hindi (all or subset) | IndicTTS Hindi + Rasa Hindi subset (command/conversational styles) + SPICOR English if access arrives | Studio recordings, human transcripts. HF mirror (SPRINGLab/IndicTTS-Hindi) is ungated but subject to the "License For Use of Indic TTS" (IIT Madras), not plain CC BY. SPRINGLab has no English subset → Indian English candidate: SPICOR TTS_1.0 (IISc SPIRE, human quality-checked, form access). Fits ~10–20 h budget. | 2026-11-04 |
| 2 | Tokenizer / script | Devanagari+Latin; Roman only; both | Devanagari (Hindi) + Latin (English), new SentencePiece | Matches dataset script, so no lossy transliteration step. Email allows either script. Cost: Roman-Hindi input is out of scope. | 2026-11-04 |
| 3 | Model size / recipe | 24l only; 24l + distill to 6l | 24l teacher, finetune_language recipe, no distillation | $30 / ~5 h budget only covers one training run; distillation listed as future work. | 2026-11-04 |
| 4 | Final checkpoint | step 4000 EMA vs step 5000 EMA (CER shortlist) | step 4000 EMA | Best held-out CER (Hindi 0.294 / 0.341, English 0.564) and preferred on listening; later steps overfit. | 2026-11-05 |

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

## Laptop smoke test (2026-11-04, RTX 4060 Laptop 8 GB)
- First attempt OOM'd in the one-time Mimi latent precompute (24l model already on GPU). Fix: run `training.scripts.precompute_latents --batch-size 4` separately first.
- 30 steps, micro-batch 4 x 16 accumulation (effective 64), 8-bit AdamW, CPU EMA: peak 5.6 GiB, 0.30–0.36 steps/s (~3 s/step). Loss ~0.48 and grad norm ~0.75 from the start, in line with the upstream README's expectations.
- Implication: 3,000 steps ≈ 2.5 h; 2,000 steps ≈ 1.7 h.

## Evaluation setup
- CER via greedy CTC wav2vec2: Hindi `theainerd/Wav2Vec2-large-xlsr-hindi`, English `facebook/wav2vec2-base-960h`. 10 held-out validation sentences per source, voice prompt `voices/indictts_hi_female.wav`.
- ASR floor on the real recordings: IndicTTS-hi 0.179, Rasa-hi 0.156, LibriTTS-en 0.010.
- Base model (released English Pocket TTS, 3 sentences/source sanity run): Hindi ~0.67–0.69 (unintelligible), English 0.006.
- Note: EMA decay 0.999 has a ~1,000-step horizon, so early EMA checkpoints still carry much of the random-init text embedding; evaluate raw (`#raw`) weights too.

## Training run `hinglish_24l_local` (laptop, launched 2026-11-04 21:23)
- Data: IndicTTS-hi 2.2 h + Rasa-hi 7.6 h + LibriTTS-R-en 2.5 h = 8,474 train clips; tokenizer 4,000 pieces over all transcripts.
- ~0.45 steps/s (3,000 steps ≈ 1.9 h; ~0.38 while CPU evals run). Valid loss: 0.422 @250, 0.391 @500, 0.337 @750.

| Step | Weights | IndicTTS-hi CER | Rasa-hi CER | LibriTTS-en CER | Notes |
| --- | --- | --- | --- | --- | --- |
| — | real recordings (ASR floor) | 0.179 | 0.156 | 0.010 | |
| — | base English 6L (released) | ~0.67 | ~0.69 | 0.006 | 3 sentences/source |
| 500 | raw | 0.640 | 0.649 | 0.771 | speech-like, not yet following text; English collapsed with the new tokenizer |
| 1000 | raw | 0.482 | 0.552 | 0.731 | Hindi words start to come through (partial); English still unintelligible |
| 1500 | raw | 0.361 | 0.438 | 0.707 | Hindi mostly recognisable; English barely moving (new Latin pieces undertrained, English ~20% of data) |
| 2000 | raw / EMA | 0.372 / 0.458 | 0.399 / 0.487 | 0.660 / 0.732 | EMA still lagging |
| 3000 | raw / EMA | 0.398 / 0.350 | 0.508 / 0.346 | 0.653 / 0.636 | EMA overtakes raw |
| 4000 | raw / EMA | 0.345 / **0.294** | 0.421 / **0.341** | 0.692 / **0.564** | best overall (EMA) |
| 5000 | raw / EMA | 0.490 / 0.289 | 0.399 / 0.356 | 0.678 / 0.682 | Hindi flat, English worse |
| 6000 | raw / EMA | 0.363 / 0.337 | 0.492 / 0.386 | 0.732 / 0.638 | worse: overfitting (~45 passes over 12 h) |
- Dataset uploaded (private): zxnjxd/pocket-tts-finetuned-data, 8,564 clips + manifests + card.

## Unattended schedule (started 2026-11-04 22:51)
`scripts/run_extended_training.sh` (run under `systemd-inhibit`, detached): waits for the 3,000-step job, resumes the same run to 6,000 steps, and scores every 1,000-step checkpoint (raw + EMA) into `outputs/eval/summary.tsv`. Every scored checkpoint is hard-linked into `runs/hinglish_24l_local/keep/`.

### Finishing by hand
1. Progress: `tail runs/extended.log`, `cat outputs/eval/summary.tsv`.
2. Pick the step with the lowest Hindi CER, then listen to its clips in `outputs/eval/step<N>_raw/` vs `_ema/`.
3. After-clips for the video:
   `.venv-gpu/bin/python -m hinglish_tts.infer --config configs/inference_local.yaml --checkpoint runs/hinglish_24l_local/keep/checkpoint_0000<N>.pt --voice voices/indictts_hi_female.wav --sentences samples/baseline_sentences.txt --out-dir samples/finetuned --prefix finetuned_`
4. Upload the model (private): `.venv-gpu/bin/python -m hinglish_tts.package_hf runs/hinglish_24l_local/keep/checkpoint_0000<N>.pt --repo zxnjxd/pocket-tts-finetuned --upload` (add `--raw` if raw weights won).
5. Check it loads from the Hub: `.venv-gpu/bin/pocket-tts generate --config hf://zxnjxd/pocket-tts-finetuned/config.yaml --voice voices/indictts_hi_female.wav --text "आपका appointment confirm हो गया है।"`
6. Fill RESULTS_TABLE / APPROACH / LIMITATIONS in README.md and docs/model_card.md, create a fine-grained read-only token for the two repos, record the video, send the email.

- Finalists for the listening check: step 4000 EMA and step 5000 EMA (`outputs/finalists/`). Extending past 6,000 steps would not help: held-out CER rises after 4,000–5,000 steps.
