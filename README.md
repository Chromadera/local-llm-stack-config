# Local LLM Stack Configuration (ROCm / RX 7900 XTX)

My complete local LLM inference stack configuration — everything runs on one machine,
no cloud APIs. Documented so a friend with the same GPU (AMD RX 7900 XTX, 24 GiB) can
replicate the setup.

## Hardware

| Component | Detail |
|---|---|
| GPU | AMD Radeon RX 7900 XTX, 24 GiB VRAM (gfx1100) |
| ROCm | 7.14.0 (`/opt/rocm`, core at `/opt/rocm/core-7.14`) |
| CPU | Ryzen 7 7800X3D |
| RAM | 64 GiB system |

## Stack

- **Runtime:** llama.cpp fork **TheTom/llama-cpp-turboquant** (TurboQuant+ codec stack),
  branch `feature/turboquant-kv-cache`
  - Turbo2 KV cache (2-bit V) + in-file MTP speculative decoding = 200K context on 24 GiB
- **Router:** llama-swap on `:8080` (single OpenAI-compatible endpoint for everything)
- **Secondary engine:** quimmedes/cafe-llama.cpp (MoE host-RAM offload, Flash-Next)
- **Also tested:** hipfire (RDNA-native Rust engine), DFlash2 spec-decode (llama.cpp PR)

## The flagship setup: Qwen3.8-27B with MTP

The dense 27B (qwen35 arch) runs at **200K context** with **MTP spec-decode** (n=4) and
**turbo2 KV** — the MTP head is in-file, so no separate draft model is needed. Four
llama-swap entries pin four reasoning-effort levels (low/medium/high/max).

Key flags:
```bash
llama-server-turbo -ngl 99 --device ROCm0 \
  --cache-type-k q8_0 --cache-type-v turbo2 --jinja --parallel 1 \
  -m Qwen3.8-27B-Q5-XYZ-v2.gguf \
  --mmproj mmproj-Qwen3.8-27B-bf16.gguf --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --chat-template-file quimmedes-chat-template.jinja \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-budget 4096 -c 200000
```

Measured: ~35 tok/s raw, ~70 tok/s accepted (MTP), 16.7 GiB VRAM at 200K ctx.

## Contents

- **`turboquant_full_config_handoff.md`** — the full write-up: build flags, env,
  all 10 model serve commands, measured performance, the hipfire + DFlash2
  experiments, and the ROCm HIP-graph wedge warning.
- **`turboquant_config.yaml`** — the verbatim llama-swap `config.yaml` as running.

## Models (all local)

| Model | Engine | Context | Spec-decode | Notes |
|---|---|---|---|---|
| Qwen3.8-27B-Q5-XYZ-v2 | turboquant | 200K | MTP n4 | flagship; 4 effort levels |
| Qwen3.8-27B-Q5-XYZ-v2 | cafe | 200K | none | live per-request effort |
| Qwen3.6-35B-A3B MoE (IQ4_NL) | turboquant | 128K | none | fast agentic coding |
| Qwen3.8-Flash-Next (28-shard) | cafe | 200K | none | MoE offload to host RAM |
| LFM2.5-2.6B Q6_K | turboquant | 128K | none | small reasoning/agent |
| LFM2.5-VL-3B Q6_K | turboquant | 128K | none | local vision |
| Qwen3-VL-8B Q4_K_M | turboquant | 32K | none | vision QA |

## Decode speed vs context depth (Qwen3.8-27B, turboquant + MTP n4)

Measured on the RX 7900 XTX via server `/metrics` counter deltas (method:
`references/long-context-decode-measurement.md`). **Context depth matters — decode
collapses ~4x from short context to 100K+.**

| Context depth | decode tok/s | prefill tok/s | notes |
|---|---|---|---|
| ~30 (short) | **75.5** | 224 | llama-bench style |
| ~few hundred (agent-style) | **~85** | ~190 | typical dsh/agent turn |
| 106K | **18.45** | 420 | pathological single-corpus dump; model FITS at 106K/24GB — it's speed, not capacity, that degrades |

The 18.45 t/s figure is the worst case (a single ~106K-token codebase in one prompt then
generating). Real agent turns at a few hundred tokens decode ~85 t/s. Quote the number for
the depth you actually run.

## Updates

This repo is updated whenever the local config changes, after a go-ahead. The
`turboquant_config.yaml` is the source of truth for the running llama-swap config;
the handoff doc is the human-readable explanation.
