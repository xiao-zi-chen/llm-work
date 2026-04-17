<p align="center">
  <img src="./.github/assets/llm-work-hero.png" alt="LLM Work hero banner" width="100%">
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&pause=1300&color=7DD3FC&center=true&vCenter=true&width=980&lines=LLM+Work+%2F%2F+PyTorch+Machine+Translation;Transformer+Variants+%2F%2F+MHA+MQA+GQA+MoE;Coursework+Code+Release+with+Clean+Open-Source+Presentation" alt="LLM Work typing intro" />
</p>

<p align="center">
  <strong>English</strong>
</p>

<p align="center">
  <strong>LLM Work presents a compact research-style machine translation project:</strong>
  PyTorch Transformer training, attention-variant comparison, tokenization pipelines, BLEU evaluation, and a lightweight knowledge editing demo.
</p>

<p align="center">
  <strong>Built for clean public release:</strong>
  the repository keeps reusable code and documentation while excluding reports, figures, and heavy training artifacts.
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/QUICK_START-5_MIN-0EA5E9?style=for-the-badge" alt="Quick Start"></a>
  <a href="#model-architecture"><img src="https://img.shields.io/badge/MODEL-ARCHITECTURE-F472B6?style=for-the-badge" alt="Model Architecture"></a>
  <a href="#training--evaluation"><img src="https://img.shields.io/badge/TRAINING-EVALUATION-84CC16?style=for-the-badge" alt="Training and Evaluation"></a>
  <a href="#code-organization"><img src="https://img.shields.io/badge/CODE-ORGANIZATION-F59E0B?style=for-the-badge" alt="Code Organization"></a>
  <a href="#project-notes"><img src="https://img.shields.io/badge/PROJECT-NOTES-334155?style=for-the-badge" alt="Project Notes"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Transformer-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch Transformer">
  <img src="https://img.shields.io/badge/datasets-HuggingFace-FCC624" alt="Hugging Face Datasets">
  <img src="https://img.shields.io/badge/tokenizers-word%20%2B%20BPE-14B8A6" alt="Word and BPE tokenizers">
  <img src="https://img.shields.io/badge/attention-MHA%20%7C%20MQA%20%7C%20GQA-0EA5E9" alt="Attention variants">
  <img src="https://img.shields.io/badge/MoE-top--k%20routing-8B5CF6" alt="MoE routing">
  <img src="https://img.shields.io/badge/entrypoints-3-3B82F6" alt="3 entrypoints">
  <img src="https://img.shields.io/badge/core_modules-7-F59E0B" alt="7 core modules">
</p>

<a id="model-architecture"></a>
## 🧠✨ Model Architecture \(^_^)/ 

<p align="center">
  <strong>🧩 one encoder-decoder core • 👀 configurable attention surfaces • 🚀 sparse routing as an optional extension</strong>
</p>

This project centers on a compact Transformer-based machine translation system with a clear comparison axis across attention mechanisms and feed-forward routing strategies.

What defines the model surface:

- 👀 **attention is configurable**: `MHA`, `MQA`, and `GQA` are exposed through one shared attention implementation
- 🧠 **the core stays readable**: encoder, decoder, positional encoding, masking, and decoding are implemented directly in PyTorch
- 🚀 **MoE is optional**: the dense FFN can be swapped for a token-level top-k Mixture-of-Experts block
- 🧪 **comparison is easy**: the same training loop and dataset pipeline can be reused across model variants

At the system surface, this repository includes:

- 🛠️ **1 training pipeline** in [src/train.py](./src/train.py)
- 📦 **1 model stack** in [src/model.py](./src/model.py)
- 🧾 **2 tokenizer paths** in [src/data.py](./src/data.py): word-level and BPE
- 🎛️ **3 attention modes**: `MHA`, `MQA`, `GQA`
- 🌐 **3 dataset options**: `Multi30k`, `OPUS-100`, `OPUS Books`
- 🤖 **1 knowledge edit demo** in [src/knowledge_edit_demo.py](./src/knowledge_edit_demo.py)

In short: the project is not just a single translation script, but a **small, inspectable, and comparison-friendly Transformer MT workspace** `(^_^)`

---

<a id="quick-start"></a>
## 🚀 Quick Start \(^o^)/

### 🧰 What You Need

| Item | Why It Matters |
| --- | --- |
| `Python 3.10+` | required for training, decoding, dataset loading, and tokenizer building |
| `pip` | enough for installing the lightweight dependency set |

### 1. 🔍 Install Dependencies

```bash
git clone https://github.com/xiao-zi-chen/llm-work.git
cd llm-work
pip install -r requirements.txt
```

### 2. 🚀 Run a Quick Experiment Suite

```bash
python src/run_experiments.py --preset quick
```

This runs a small comparison preset and writes local outputs under `outputs/`.

### 3. 🧪 Run a Smoke Test

```bash
python src/run_experiments.py --preset smoke
```

This is the fastest way to verify that the full pipeline works end to end.

### 4. 🛠️ Train a Single Translation Model

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

### 🧭 Try a Stronger Configuration

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

### 💬 Main Entry Points

```text
src/train.py
src/run_experiments.py
src/knowledge_edit_demo.py
```

---

<a id="training--evaluation"></a>
## 🛰️📊 Training & Evaluation \(^o^)/

<p align="center">
  <strong>📚 dataset loading • 🧾 tokenizer training • 📈 optimization and BLEU • 🔬 reproducible preset runs</strong>
</p>

The repository includes the full path from raw parallel text to decoded samples and metric files.

What the pipeline covers:

- 📚 loads translation pairs from Hugging Face datasets
- 🧾 trains tokenizers and saves them alongside model outputs
- 📈 trains with AdamW, Noam scheduling, label smoothing, and checkpoint selection by validation loss
- 🔍 evaluates with greedy decoding or beam search
- 🌍 reports corpus BLEU with `sacrebleu`

### ✨ Supported Comparison Axes

| Axis | Options | Why It Matters |
| --- | --- | --- |
| attention | `mha`, `mqa`, `gqa` | compare standard, shared-KV, and grouped-KV attention |
| feed-forward | dense, `--use-moe` | compare standard FFN with sparse top-k routing |
| tokenizer | `word`, `bpe` | compare simple vocabulary baselines and subword modeling |
| dataset | `multi30k`, `opus100`, `opus_books` | test across different scales and language pairs |
| decoding | greedy, beam search | trade off speed and translation quality |

### 🧪 Preset Runner

The preset launcher in [src/run_experiments.py](./src/run_experiments.py) supports:

- `smoke`: fastest sanity check
- `quick`: small but meaningful comparison runs
- `full`: larger coursework-scale runs

Generated outputs such as checkpoints, tokenizers, metrics files, and sample translations stay in local `outputs/` and are intentionally excluded from version control.

---

<a id="code-organization"></a>
## 🧩🗂️ Code Organization (•‿•)

<p align="center">
  <strong>👀 small public code release • 🛠️ focused module boundaries • 📦 no report clutter in the tracked repository</strong>
</p>

If you want to understand the project quickly, start here:

- 🚀 read [src/model.py](./src/model.py) for the Transformer, attention, and MoE definitions
- 🧭 read [src/data.py](./src/data.py) for dataset loading, normalization, and tokenizer logic
- 🧠 read [src/train.py](./src/train.py) for training, validation, BLEU, and decoding
- 📜 read [src/run_experiments.py](./src/run_experiments.py) for preset experiment orchestration
- 🔬 read [src/knowledge_edit_demo.py](./src/knowledge_edit_demo.py) for the toy rank-one edit demo

### 📦 Public Repository Layout

```text
.
|-- .github/
|   `-- assets/
|       `-- llm-work-hero.png
|-- docs/
|   `-- image_prompt.md
|-- requirements.txt
|-- src/
|   |-- __init__.py
|   |-- data.py
|   |-- knowledge_edit_demo.py
|   |-- model.py
|   |-- run_experiments.py
|   |-- train.py
|   `-- utils.py
`-- README.md
```

### 🧾 Module Guide

| File | Role |
| --- | --- |
| [src/model.py](./src/model.py) | Transformer encoder-decoder, MHA/MQA/GQA attention, dense FFN, MoE FFN |
| [src/data.py](./src/data.py) | dataset loading, normalization, tokenizers, dataloaders |
| [src/train.py](./src/train.py) | training CLI, loss computation, evaluation, decoding, checkpointing |
| [src/run_experiments.py](./src/run_experiments.py) | experiment matrix and preset execution |
| [src/utils.py](./src/utils.py) | seeds, scheduler, JSON helpers, counters, device helpers |
| [src/knowledge_edit_demo.py](./src/knowledge_edit_demo.py) | toy knowledge editing example with local rank-one update |

---

<a id="project-notes"></a>
## 🎛️📝 Project Notes (-_-)

<p align="center">
  <strong>🧹 code-first release • 🎓 coursework origin • 📉 reports and heavy outputs removed from version control</strong>
</p>

This public repository is intentionally narrower than the original working directory.

What is included:

- ✅ reusable source code
- ✅ dependency file
- ✅ presentation-grade README and hero image

What is excluded:

- 🛑 report documents
- 🛑 figure exports used only for submission materials
- 🛑 trained checkpoints and other heavy generated artifacts

The goal is to keep the repository clean, fast to browse, and suitable as a portfolio-style code release.

---

## 📄 License

No license file has been added yet. If you want, I can add an open-source license next.
