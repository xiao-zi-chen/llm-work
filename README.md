# LLM Work

<div align="center">

English-French / English-German machine translation experiments built with PyTorch Transformer variants.

[Quick Start](#quick-start) | [Project Highlights](#project-highlights) | [Project Structure](#project-structure) | [Experiments](#experiments)

</div>

## Overview

This repository contains a compact machine translation project for coursework and self-study. It implements a full training and evaluation pipeline around Transformer-based sequence-to-sequence models, with a focus on comparing different attention designs and lightweight sparse feed-forward routing.

The codebase includes:

- A baseline Transformer encoder-decoder for neural machine translation
- Unified support for `MHA`, `MQA`, and `GQA` through configurable KV heads
- Optional token-level `MoE` feed-forward blocks with auxiliary balancing loss
- Word-level and BPE tokenization pipelines
- Dataset loaders for `Multi30k`, `OPUS-100`, and `OPUS Books`
- Reproducible training, BLEU evaluation, decoding, and experiment runner scripts
- A small knowledge editing demo showing a local rank-one edit idea in PyTorch

## Project Highlights

### 1. Configurable Attention Variants

The implementation exposes multiple attention styles behind one module:

- `MHA`: standard multi-head attention
- `MQA`: multi-query attention with shared key/value heads
- `GQA`: grouped-query attention as a middle ground between MHA and MQA

This makes the repository useful for controlled architecture comparisons without changing the overall training pipeline.

### 2. Lightweight MoE Feed-Forward Layer

The standard Transformer FFN can be replaced with a token-level Mixture-of-Experts block:

- Top-k routing
- Multiple expert FFNs
- Simple load-balancing auxiliary loss

This keeps the code short and readable while still demonstrating a sparse routing mechanism.

### 3. End-to-End Training Pipeline

The repository is not only a model definition dump. It also includes:

- Dataset loading and split handling
- Tokenizer training and saving
- Training / validation loops
- BLEU evaluation with greedy or beam decoding
- Preset experiment launcher for quick and full runs

## Quick Start

### Environment

```bash
pip install -r requirements.txt
```

### Run a quick experiment suite

```bash
python src/run_experiments.py --preset quick
```

### Run a smoke test

```bash
python src/run_experiments.py --preset smoke
```

### Train a single model

```bash
python src/train.py \
  --dataset multi30k \
  --src-lang en \
  --tgt-lang de \
  --tokenizer word \
  --attn-type mha \
  --epochs 3 \
  --train-size 4000 \
  --valid-size 500 \
  --test-size 300
```

### Try a more advanced setting

```bash
python src/train.py \
  --dataset opus100 \
  --dataset-config en-fr \
  --src-lang en \
  --tgt-lang fr \
  --tokenizer bpe \
  --vocab-size 8000 \
  --attn-type gqa \
  --num-kv-heads 2 \
  --use-moe \
  --epochs 3 \
  --train-size 8000 \
  --valid-size 800 \
  --test-size 400
```

## Project Structure

```text
.
|-- requirements.txt
|-- src
|   |-- data.py
|   |-- knowledge_edit_demo.py
|   |-- model.py
|   |-- run_experiments.py
|   |-- train.py
|   `-- utils.py
`-- README.md
```

## Core Components

### `src/model.py`

Defines the modeling stack:

- Sinusoidal positional encoding
- Configurable attention block for MHA / MQA / GQA
- Dense FFN and sparse MoE FFN
- Transformer encoder-decoder model

### `src/data.py`

Handles:

- Dataset loading from Hugging Face datasets
- Text normalization
- Word tokenizer and BPE tokenizer training
- Batch collation and dataloader creation

### `src/train.py`

Provides:

- CLI training interface
- Validation and BLEU evaluation
- Greedy decoding and beam search decoding
- Checkpoint saving and metrics export

### `src/run_experiments.py`

Runs predefined experiment suites for:

- smoke testing
- quick coursework-scale comparisons
- larger full runs

### `src/knowledge_edit_demo.py`

Implements a concise demo of a local orthogonal rank-one edit for factual memory manipulation in a toy setting.

## Experiments

The experiment runner supports three presets:

- `smoke`: fastest sanity check
- `quick`: small but meaningful comparison runs
- `full`: larger training configurations

Outputs are written under `outputs/` during local runs and include:

- model checkpoints
- tokenizers
- `metrics.json`
- decoded translation samples

These generated artifacts are intentionally excluded from version control in this public repository.

## Notes

- The project is designed to be compact and readable rather than heavily optimized.
- Default settings are sized to make coursework reproduction easier on limited hardware.
- BLEU and runtime will vary with GPU availability, tokenizer choice, dataset size, and decoding strategy.

## Acknowledgement

This repository is organized as a clean code release derived from a course project, with the public version focused on reusable implementation code rather than report materials.
