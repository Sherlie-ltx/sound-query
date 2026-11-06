# Sound Query

**Audio retrieval with perceptual text modifiers.**

Sound Query asks: *given a sound and a word like "warmer" or "more spacious", can we find the version of that sound that matches the description?*

The query combines a reference audio clip with a perceptual text modifier ("brighter", "more lo-fi", "more nostalgic"). The system retrieves the target audio that best matches both. It builds on frozen [CLAP](https://huggingface.co/laion/clap-htsat-unfused) audio–text embeddings and adds a lightweight **Modifier Adapter** trained to move an audio embedding in the direction a modifier describes.

> MSc dissertation project, Sound and Music Computing (AI and Music Data Science), Queen Mary University of London.
> Supervisor: Dr. Anna Xambó Sedó. Awarded a distinction.

---

## Overview

```
reference audio ──► CLAP audio encoder ──► source embedding ─┐
                                                              ├─► Modifier Adapter ──► query ──► rank target audio
"more dreamy"   ──► CLAP text encoder  ──► modifier embedding ┘
```

1. **Dataset construction.** Each [ESC-50](https://github.com/karolpiczak/ESC-50) clip is processed with DSP effect chains, one per modifier. This gives `(source audio, modifier text, target audio)` triplets.
2. **Embedding.** Source and target audio are encoded with CLAP (`laion/clap-htsat-unfused`, 48 kHz). Each modifier's text embedding is the averaged embedding of five prompt templates (e.g. *"a sound that is warmer"*).
3. **Baselines.** Zero-shot retrieval with audio-only, text-only, and weighted audio+text query embeddings.
4. **Modifier Adapter.** A small residual MLP that predicts how the source embedding should shift, trained with a symmetric contrastive (InfoNCE) loss.
5. **Evaluation.** Global retrieval metrics (R@K, MRR, median rank), within-source modifier discrimination with confusion matrices, and stimuli generation for a human listening test.

---

## Perceptual Modifiers

The 11 modifiers are grouped by abstraction level. Each one is realised as a reproducible DSP recipe.

| Level | Modifier | Effect recipe |
|---|---|---|
| L1 – low-level | brighter | high-shelf EQ boost |
| | more muffled | low-pass filter + high cut |
| | more distorted | tanh saturation |
| | more spacious | synthetic reverb |
| | less muddy | low-mid cut + slight high boost |
| L2 – mid-level | more cinematic | low boost, low-mid cut, compression, reverb, soft high cut |
| | more dreamy | chorus + reverb + soft high cut |
| | more lo-fi | band-pass, saturation, bitcrush, noise |
| | warmer | low-mid boost, high cut, mild saturation |
| L3 – high-level / affective | more tense | high-mid boost, compression, saturation, tremolo |
| | more nostalgic | warm EQ, high cut, tape saturation, wow & flutter, noise, small reverb |

All effects (biquad shelving/peaking EQ, Butterworth filters, reverb, chorus, compression, tremolo, wow/flutter, bitcrush) are implemented from scratch with NumPy/SciPy. Random components are seeded per file and modifier, so the dataset is deterministic.

The 9-modifier set (L1 + L2) can be selected with `--modifier-set 9`.

---

## Modifier Adapter

```
x = [s, m, s ⊙ m, s − m]          # s: source emb, m: modifier emb (512-d each)
h = LayerNorm(x)
h = GELU(Linear 2048→1024) → Dropout
h = GELU(Linear 1024→1024) → Dropout
δ = Linear 1024→512               # zero-initialised
query = normalize(s + δ)
```

The final layer is zero-initialised, so training starts from the audio-only baseline and learns a residual edit. CLAP itself stays frozen, so only the adapter's parameters are trained on precomputed embeddings.

**Training defaults:** AdamW (lr 1e-3, weight decay 1e-4), batch size 512, temperature 0.07, gradient clipping 1.0, up to 80 epochs, early stopping on validation R@1 (patience 12).

**Splits** follow the ESC-50 folds: folds 1–3 → train, fold 4 → validation, fold 5 → test. Source clips never appear in more than one split.

---

## Installation

```bash
git clone https://github.com/<your-username>/sound-query.git
cd sound-query
pip install -r requirements.txt
```

Python 3.9+ is recommended. A GPU speeds up the CLAP embedding step but is not required for training the adapter.

Download ESC-50 from the [official repository](https://github.com/karolpiczak/ESC-50). The script accepts either the extracted folder or the zip file.

---

## Usage

The whole pipeline is a single script with subcommands:

```bash
# 1. Build the triplet dataset (11 modifiers)
python project.py build-triplets \
    --esc50-zip path/to/ESC-50-master.zip \
    --output-root data/sound_query \
    --modifier-set 11 \
    --copy-originals

# 2. Precompute CLAP embeddings
python project.py embed --dataset-root data/sound_query

# 3. Zero-shot baselines
python project.py evaluate-baselines --dataset-root data/sound_query

# 4. Train and evaluate the Modifier Adapter
python project.py train-adapter --dataset-root data/sound_query

# 5. Within-source evaluation + confusion matrices
python project.py evaluate-within-source \
    --dataset-root data/sound_query \
    --checkpoint-path data/sound_query/modifier_adapter_11mod_best.pt

# 6. Generate listening-test stimuli
python project.py make-listening-stimuli \
    --dataset-root data/sound_query \
    --checkpoint-path data/sound_query/modifier_adapter_11mod_best.pt \
    --output-dir listening_test

# 7. Plot result figures
python project.py plot-results --dataset-root data/sound_query
```

Run `python project.py <command> --help` to see all options.

---

## Evaluation

**Global retrieval:** for each `(source, modifier)` query, all target clips in the same split are ranked by cosine similarity. Reported metrics are R@1/5/10/50, MRR, and median/mean rank, both overall and per modifier.

**Within-source retrieval:** candidates are restricted to the 11 processed versions of the same source clip. This isolates whether the system understands the modifier itself, separately from recognising the source sound. Results include a modifier confusion matrix.

**Listening test:** the `make-listening-stimuli` command selects reference clips from the test set, retrieves the adapter's top-1 result for each modifier, and loudness-normalises the audio. It also exports a build sheet for an online form, in which participants rate (1) similarity to the reference and (2) match with the modifier on a 1–5 scale. An HTML preview page is generated as well.

Compared methods: `audio_only`, `text_only`, weighted CLAP fusions (e.g. `weighted_065_035`), and `modifier_adapter`.

## Results

<!-- Add your key numbers and figures here, e.g.: -->
<!-- ![Overall test retrieval](figures/fig_overall_test_retrieval_r_at_k.png) -->
<!-- ![R@1 by modifier](figures/fig_adapter_test_r1_by_modifier.png) -->

*Results table and figures to be added.*

---

## Output Files

| Step | Main outputs |
|---|---|
| `build-triplets` | `audio_processed/`, `metadata_triplets_11mod.csv`, `effect_recipes.json` |
| `embed` | `clap_embeddings_11mod.pt`, `metadata_11mod_with_embedding_indices.csv` |
| `evaluate-baselines` | `zeroshot_baseline_summary.csv`, `..._by_modifier.csv`, `..._ranks.csv` |
| `train-adapter` | `modifier_adapter_11mod_best.pt`, train log, eval summary CSVs |
| `evaluate-within-source` | predictions, summaries, confusion matrix CSV/PNG |
| `make-listening-stimuli` | normalised audio, trial CSVs, form build sheet, `preview.html` |
| `plot-results` | `paper_ready_results/` figures and tables |

---

## Acknowledgements & Data

- **ESC-50**: K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015. ESC-50 is released under CC BY-NC 3.0, so it is not redistributed in this repository.
- **CLAP**: Y. Wu et al., *Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation*, ICASSP 2023. Model weights via Hugging Face (`laion/clap-htsat-unfused`).
- Thanks to Dr. Anna Xambó Sedó for supervision.

## Author

**Tianxue (Shirley) Li**: MSc Sound and Music Computing, Queen Mary University of London; BM Electronic Production and Design, Berklee College of Music.
