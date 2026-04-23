# OpenMythos

[![PyPI](https://img.shields.io/pypi/v/open-mythos?style=for-the-badge&color=3670A0)](https://pypi.org/project/open-mythos/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange?style=for-the-badge)]()

An open-source theoretical reconstruction of the **Recurrent-Depth Transformer (RDT)** — a speculative architecture hypothesized to underlie Claude Mythos. OpenMythos explores how looping a compact set of transformer layers enables deeper reasoning through inference-time compute scaling, rather than relying on raw parameter count.

> **Disclaimer:** OpenMythos is an independent, community-driven project based solely on publicly available research and speculation. It is not affiliated with, endorsed by, or connected to Anthropic or any of their proprietary systems.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Variants](#model-variants)
- [Training](#training)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

---

## Overview

Standard transformers scale depth by stacking unique layers. OpenMythos takes a different approach: a single **RecurrentBlock** runs in a loop for `T` iterations at inference time. This enables:

- **Inference-time compute scaling** — run more loops on harder problems without retraining
- **Depth extrapolation** — train on N loops, test on N+k loops for improved reasoning
- **Parameter efficiency** — achieve quality comparable to a deeper fixed model with far fewer unique parameters
- **Implicit chain-of-thought** — each loop iteration is a reasoning step in continuous latent space

---

## Architecture

The model has three stages:

```
Input Tokens
    │
    ▼
┌─────────────┐
│   Prelude   │  — Standard transformer layers (run once)
└─────────────┘
    │
    ▼
┌─────────────┐
│  Recurrent  │  — Single transformer block, looped T times
│    Block    │    with MoE FFN, LoRA, LTI injection, ACT halting
└─────────────┘  ◄─── loops back T times
    │
    ▼
┌─────────────┐
│    Coda     │  — Standard transformer layers (run once)
└─────────────┘
    │
    ▼
  Logits
```

### Key Components

| Component | Role |
|---|---|
| **GQA / MLA** | Grouped Query Attention or Multi-Latent Attention (10-20x smaller KV cache) |
| **MoE FFN** | Mixture-of-Experts with top-K routed + always-on shared experts |
| **LoRA Adapter** | Depth-wise LoRA for per-loop weight differentiation without parameter explosion |
| **LTI Injection** | Stable recurrent state update: `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` |
| **ACT Halting** | Adaptive Computation Time — per-position early exit to avoid overthinking |
| **RoPE** | Rotary Position Embeddings for positional encoding |

### Stability Guarantee

The `LTIInjection` module parameterizes `A` as a diagonal matrix with negative values, constrained so the spectral radius `ρ(A) < 1` always holds. This prevents hidden state divergence regardless of loop depth.

---

## Installation

```bash
# From PyPI
pip install open-mythos

# With Flash Attention 2 (requires CUDA and build tools)
pip install open-mythos[flash]

# From source
git clone https://github.com/The-Swarm-Corporation/OpenMythos
cd OpenMythos
pip install poetry && poetry install
```

**Requirements:** Python 3.10+, PyTorch 2.1+, Transformers 4.40+

---

## Quick Start

### GQA Attention

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=32000,
    dim=256,
    n_heads=8,
    n_kv_heads=2,          # GQA: fewer KV heads than Q heads
    max_seq_len=512,
    max_loop_iters=8,
    prelude_layers=2,
    coda_layers=2,
    n_experts=16,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=128,
    lora_rank=8,
    attn_type="gqa",
)

model = OpenMythos(cfg)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

tokens = torch.randint(0, cfg.vocab_size, (1, 64))
logits = model(tokens, n_loops=8)       # n_loops can exceed training depth
output = model.generate(tokens, max_new_tokens=32, n_loops=12)
```

### MLA Attention (compressed KV cache)

```python
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=32000,
    dim=512,
    n_heads=8,
    n_kv_heads=8,
    max_seq_len=1024,
    max_loop_iters=8,
    prelude_layers=2,
    coda_layers=2,
    n_experts=16,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=128,
    lora_rank=8,
    attn_type="mla",
    kv_lora_rank=64,       # compressed KV latent dimension
    q_lora_rank=128,
    qk_rope_head_dim=32,
    qk_nope_head_dim=32,
    v_head_dim=32,
)

model = OpenMythos(cfg)
```

### Using Pre-configured Variants

```python
from open_mythos import mythos_1b, mythos_3b, OpenMythos

cfg = mythos_3b()
model = OpenMythos(cfg)
```

### Checking Stability

```python
A = model.recurrent.injection.get_A()
rho = torch.linalg.eigvals(A).abs().max().item()
print(f"Spectral radius ρ(A) = {rho:.4f}  (must be < 1.0)")
```

---

## Model Variants

Pre-configured factory functions for common scales:

| Variant | Parameters | Context | Hidden | Experts | Loops | Use Case |
|---|---|---|---|---|---|---|
| `mythos_1b()` | ~1B | 4k | 2048 | 64 | 16 | Research / fine-tuning |
| `mythos_3b()` | ~3B | 4k | 3072 | 64 | 16 | Compact inference |
| `mythos_10b()` | ~10B | 8k | 4096 | 128 | 24 | Mid-scale general |
| `mythos_50b()` | ~50B | 8k | 6144 | 256 | 32 | Large reasoning |
| `mythos_100b()` | ~100B | 1M | 8192 | 256 | 32 | Frontier-class |
| `mythos_500b()` | ~500B | 1M | 12288 | 512 | 48 | Ultra-scale MoE |
| `mythos_1t()` | ~1T | 1M | 16384 | 512 | 64 | Maximum scale |

---

## Training

### Single GPU

```bash
python training/3b_fine_web_edu.py
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
  training/3b_fine_web_edu.py
```

Training uses the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset (1.3T tokens) by default. See [docs/datasets.md](docs/datasets.md) for other recommended datasets and token budget guidance per model size.

**Training dependencies:**

```bash
pip install -r training/requirements.txt
```

---

## Testing

```bash
# Full test suite
pytest tests/

# Benchmarks
python tests/small_benchmark.py
python tests/bench_vs_transformer.py
```

Test configurations use tiny models (64-dim, 2-4 heads) that run quickly on CPU.

---

## Project Structure

```
open_mythos/
├── main.py          # Core architecture (OpenMythos, MythosConfig, all components)
├── variants.py      # Pre-configured model sizes (mythos_1b … mythos_1t)
├── tokenizer.py     # MythosTokenizer
├── moda.py          # MoDA model variant
└── __init__.py      # Public API exports

training/
├── 3b_fine_web_edu.py   # FineWeb-Edu pretraining script
└── requirements.txt

tests/
├── test_main.py         # Core architecture tests
├── test_tokenizer.py    # Tokenizer tests
├── test_rope_debug.py   # RoPE debugging
├── small_benchmark.py   # Performance benchmarks
└── bench_vs_transformer.py

docs/
├── open_mythos.md   # Full API reference
└── datasets.md      # Training dataset recommendations

examples/
├── variants_example.py
└── moda_example.py
```

---

## References

This project is informed by the following public research:

- [Universal Transformers](https://arxiv.org/abs/1807.03819) (Dehghani et al., 2019)
- [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13379) (Giannou et al., 2023)
- [Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10073) (Elbayad et al., 2020)
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) (Graves, 2016)
- [DeepSeek-V2: MLA Attention](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024)
- [Mixture of Experts](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)
- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [RoPE](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

OpenMythos is an independent research project. It is not affiliated with, endorsed by, or connected to Anthropic or Claude in any official capacity.
