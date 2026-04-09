# Energy-Based Diffusion Language Models for Text Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MinkaiXu/Energy-Diffusion-LLM/LICENSE)
[![ArXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2410.21357-B31B1B.svg)](https://arxiv.org/abs/2410.21357)

Official code release for the paper **"Energy-Based Diffusion Language Models for Text Generation"**, accepted at *The 13th International Conference on Learning Representations (ICLR), 2025*.

This repository implements Energy-Based Diffusion Language Models (EDLM) that combine diffusion models with energy-based models for high-quality text generation. The framework supports multiple backbone architectures including DiT (Diffusion Transformer), DiMamba, Autoregressive (AR), and UTRLM.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Sampling](#sampling)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Datasets](#datasets)
- [Checkpointing](#checkpointing)
- [Citation](#citation)

---

## Overview

Energy-Based Diffusion Language Models (EDLM) enhance standard diffusion language models by incorporating an energy-based model (EBM) that guides the generation process. The key innovation is using importance sampling guided by the energy function to improve sample quality.

**Key Features:**
- Multiple backbone architectures: DiT, DiMamba, AR, UTRLM
- Support for various diffusion parameterizations: SUBS, D3PM, SEDD
- Energy-based importance sampling for improved generation
- Compatible with pretrained checkpoints from MDLM
- Flexible configuration via Hydra

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git LFS (for downloading checkpoints)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd UTRLM-EDLM-main
```

2. Create and activate a conda environment:
```bash
conda create -n edlm python=3.10
conda activate edlm
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install additional dependencies (if using Mamba backbones):
```bash
pip install causal-conv1d>=1.2.0 mamba-ssm
```

---

## Quick Start

### 1. Download Pretrained Checkpoints

Download pretrained Diffusion and AR LLM checkpoints from the [MDLM repository](https://github.com/kuleshov-group/mdlm?tab=readme-ov-file#checkpoints).

Save checkpoints to the `checkpoints/` folder:
```bash
mkdir -p checkpoints
# Place downloaded checkpoints here, e.g., checkpoints/ar.ckpt
```

### 2. Run Evaluation (No Training Required)

We recommend starting with these scripts that don't require training:

```bash
# Sample evaluation with AR-EBM
bash scripts/job_sample_eval_owt_T_arebm.sh

# Perplexity evaluation with AR-EBM
bash scripts/job_eval_owt_T_arebm.sh
```

### 3. Generate Text Samples

```bash
python -m main \
    mode=sample_eval \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    sampling.steps=128 \
    sampling.num_sample_batches=2
```

---

## Project Structure

```
UTRLM-EDLM-main/
├── configs/                    # Hydra configuration files
│   ├── callbacks/             # Training callbacks (checkpointing, LR monitoring)
│   ├── data/                  # Dataset configurations
│   ├── lr_scheduler/          # Learning rate schedules
│   ├── model/                 # Model architecture configs
│   ├── noise/                 # Noise schedule configs
│   ├── strategy/              # Distributed training strategies
│   └── config.yaml            # Main configuration file
├── models/                    # Model implementations
│   ├── autoregressive.py      # AR backbone
│   ├── dimamba.py             # DiMamba backbone
│   ├── dit.py                 # DiT backbone
│   └── ema.py                 # Exponential Moving Average
├── scripts/                   # Training/evaluation scripts
│   ├── job_train_owt_ebm.sh   # Train EBM on OpenWebText
│   ├── job_train_owt_arebm.sh # Train AR-EBM
│   ├── job_sample_eval_owt_T_ebm.sh    # Sample evaluation
│   ├── job_sample_eval_owt_T_arebm.sh  # AR-EBM sampling
│   ├── job_eval_owt_T_ebm.sh  # Perplexity evaluation
│   └── job_eval_owt_T_arebm.sh
├── dataloader.py              # Data loading and preprocessing
├── diffusion.py               # Diffusion and EBM model implementations
├── main.py                    # Main entry point
├── noise_schedule.py          # Noise scheduling
├── utils.py                   # Utility functions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Usage

### Training

#### Train a Standard Diffusion Model

```bash
python -m main \
    mode=train \
    backbone=dit \
    data=openwebtext-split \
    model=small \
    loader.batch_size=16 \
    optim.lr=3e-4
```

#### Train an Energy-Based Model (EBM)

```bash
python -m main \
    mode=train \
    backbone=hf_dit \
    ebm_backbone=dit \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    data=openwebtext-split \
    model=small \
    loader.batch_size=16
```

#### Train with AR-EBM Backbone

```bash
python -m main \
    mode=train \
    backbone=hf_dit \
    ebm_backbone=ar \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    data=openwebtext-split \
    model=small
```

### Evaluation

#### Compute Perplexity

```bash
python -m main \
    mode=ppl_eval \
    eval.checkpoint_path=checkpoints/last.ckpt \
    data=openwebtext-split \
    model.length=1024
```

### Sampling

#### Generate Samples with DDPM Cache

```bash
python -m main \
    mode=sample_eval \
    eval.checkpoint_path=checkpoints/last.ckpt \
    sampling.predictor=ddpm_cache \
    sampling.steps=128 \
    sampling.num_sample_batches=10 \
    loader.eval_batch_size=1
```

#### Generate with Energy-Based Importance Sampling

```bash
python -m main \
    mode=sample_eval \
    eval.checkpoint_path=checkpoints/last.ckpt \
    sampling.predictor=ddpm_cache \
    sampling.steps=128 \
    sampling.is_size=2 \
    sampling.is_start=1.0 \
    sampling.is_end=0.4 \
    sampling.is_temp=1.0
```

---

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Key configuration files are in `configs/`.

### Main Configuration Options (`configs/config.yaml`)

| Parameter | Description | Options |
|-----------|-------------|---------|
| `mode` | Operation mode | `train`, `ppl_eval`, `sample_eval` |
| `backbone` | Diffusion backbone | `dit`, `dimamba`, `ar`, `hf_dit`, `utrlm` |
| `ebm_backbone` | EBM backbone | `dit`, `ar`, `hf_dit`, `utrlm` |
| `parameterization` | Diffusion parameterization | `subs`, `d3pm`, `sedd` |
| `diffusion` | Diffusion type | `absorbing_state` |
| `T` | Number of timesteps | `0` (continuous), `1000` (discrete) |

### Model Configurations (`configs/model/`)

- `tiny.yaml`: Small model for testing
- `small.yaml`: Standard small model
- `medium.yaml`: Medium-sized model
- `tiny-ar.yaml`: Tiny autoregressive model
- `tiny-dimamba.yaml`: Tiny Mamba-based model

### Data Configurations (`configs/data/`)

Supported datasets:
- `openwebtext.yaml`: OpenWebText corpus
- `openwebtext-split.yaml`: OpenWebText with train/val split
- `wikitext2.yaml`, `wikitext103.yaml`: WikiText datasets
- `text8.yaml`, `text8-crop.yaml`: Text8 character-level
- `lambada.yaml`: LAMBADA benchmark
- `ag_news.yaml`: AG News classification

---

## Model Architectures

### DiT (Diffusion Transformer)

Transformer-based diffusion model with rotary embeddings and flash attention.

**File**: `models/autoregressive.py` (class `DDIT`)

### DiMamba

Mamba-based diffusion model with bidirectional support.

**File**: `models/dimamba.py` (class `DiMamba`)

### AR (Autoregressive)

Standard autoregressive transformer for comparison.

**File**: `models/autoregressive.py` (class `AR`)

### UTRLM

RNA language model adapted for text generation tasks.

**File**: Integrated via `multimolecule` library

---

## Datasets

The framework supports multiple text datasets through HuggingFace's `datasets` library:

| Dataset | Description | Use Case |
|---------|-------------|----------|
| OpenWebText | Web text corpus | General pretraining |
| WikiText-2/103 | Wikipedia articles | Language modeling |
| Text8 | Character-level text | Character modeling |
| LAMBADA | Long-context understanding | Evaluation |
| AG News | News articles | Classification |

**Data Cache**: Datasets are cached to `~/.cache/huggingface/datasets/` by default.

---

## Checkpointing

### Saving Checkpoints

Checkpoints are automatically saved based on configuration in `configs/callbacks/checkpoint_every_n_steps.yaml` and `checkpoint_monitor.yaml`.

Default save location: `outputs/<experiment_name>/checkpoints/`

### Resuming Training

To resume from a checkpoint:
```bash
python -m main \
    mode=train \
    checkpointing.resume_from_ckpt=true \
    checkpointing.resume_ckpt_path=outputs/<exp>/checkpoints/last.ckpt
```

### Loading Pretrained Models

For evaluation or sampling from pretrained models:
```bash
python -m main \
    mode=sample_eval \
    eval.checkpoint_path=kuleshov-group/mdlm-owt
```

---

## Advanced Features

### Energy-Based Importance Sampling

Enable importance sampling during generation:
```yaml
sampling:
  is_size: 2          # Number of importance samples
  is_start: 1.0       # Start timestep for IS
  is_end: 0.4         # End timestep for IS
  is_temp: 1.0        # Temperature for energy weighting
```

### Semi-Autoregressive Sampling

For long sequence generation:
```yaml
sampling:
  semi_ar: true
  stride_length: 64
  num_strides: 16
```

### EMA (Exponential Moving Average)

EMA is enabled by default for stable training:
```yaml
training:
  ema: 0.9999
```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `loader.batch_size` or `model.length`
2. **Tokenizer Errors**: Ensure tokenizer files are downloaded
3. **Checkpoint Loading**: Verify checkpoint path exists

### Environment Variables

For offline mode (no HuggingFace hub access):
```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Acknowledgements

This repository is built upon the [MDLM](https://github.com/kuleshov-group/mdlm) repository. We thank the authors for their excellent work on masked diffusion language models.

Additional dependencies:
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Mamba](https://github.com/state-spaces/mamba)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{xu2024energy,
  title={Energy-based diffusion language models for text generation},
  author={Xu, Minkai and Geffner, Tomas and Kreis, Karsten and Nie, Weili and Xu, Yilun and Leskovec, Jure and Ermon, Stefano and Vahdat, Arash},
  journal={arXiv preprint arXiv:2410.21357},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License.
