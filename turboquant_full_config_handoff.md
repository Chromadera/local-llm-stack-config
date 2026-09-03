# Chromadera's Local LLM Stack — Full Config (TurboQuant+ on ROCm, RX 7900 XTX 24GB)
# Handoff for a friend with the same GPU. Everything local, no cloud.
# Last verified: 2026-09-03. Recon only — this is the running config as-is.

## 1. HARDWARE / STACK
- GPU: AMD Radeon RX 7900 XTX, 24 GiB VRAM (gfx1100). (There is a tiny second GPU[1] 512MB — ignore it.)
- ROCm: 7.14.0 (`/opt/rocm`, core at `/opt/rocm/core-7.14`)
- LLM runtime: llama.cpp fork **TheTom/llama-cpp-turboquant** (TurboQuant+ codec stack)
  - branch `feature/turboquant-kv-cache`, HEAD `c26cbdf` (PR #225)
  - https://github.com/TheTom/llama-cpp-turboquant
- Router: llama-swap (`~/llama-swap/llama-swap`, config `~/llama-swap/config.yaml`), listens :8080
- System RAM: 64 GiB (relevant: long ctx + checkpoints can approach OOM — see note at bottom)
- OS: Linux (Ubuntu-class), kernel 7.0.0-30-generic

## 2. THE BUILD (from CMakeCache — exact flags)
```
cmake -B build \
  -DGGML_HIP=ON -DGGML_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/home/sopdet/.local/rocm-staging \
  -DGPU_TARGETS=gfx1100 \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_HIP_RCCL=OFF
```
Key: HIP on, CUDA off, `gfx1100`, rocm-staging root, no VMM. Release build.

## 3. RUNTIME ENV (launcher env — REQUIRED, the ambient LD_LIBRARY_PATH shadows it)
```bash
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib/llvm/lib:/opt/rocm/core-7.14/lib/rocm_sysdeps/lib:/opt/rocm/core-7.14/lib/llvm/lib/clang/23/lib/linux:/home/sopdet/.local/rocm-staging/lib"
```
This is in `~/.local/bin/llama-turbo-env`. Without it: `undefined symbol` errors from wrong libs.

## 4. MODEL LAUNCHERS (thin wrappers that set env + exec the right binary)
- `~/.local/bin/llama-server-turbo` -> execs `/home/sopdet/llama-cpp-turboquant/build/bin/llama-server` (the turboquant build)
- `~/.local/bin/llama-server-cafe`  -> execs `/home/sopdet/cafe-llama/build/bin/llama-server`  (separate quimmedes/cafe-llama.cpp fork, see note)

## 5. THE MODELS + EXACT SERVE FLAGS (llama-swap config.yaml, all 10 entries)
All `-ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT} --jinja --parallel 1`.

### A. The flagship: Qwen3.8-27B-Q5-XYZ-v2 (dense 27B, 200K ctx, MTP spec-decode)  <- the one to replicate
4 entries = 4 reasoning-effort levels (turboquant ignores per-request effort; it's launch-pinned):

**qwen38-q5 (low)** / **qwen38-q5-med (medium)** / **qwen38-q5-hi (high)** / **qwen38-q5-max (max)**
```bash
llama-server-turbo -ngl 99 --device ROCm0 --port ${PORT} \
  --cache-type-k q8_0 --cache-type-v turbo2 --jinja --parallel 1 --metrics \
  -m /home/sopdet/models-q38/Qwen3.8-27B-Q5-XYZ-v2.gguf \
  --mmproj /home/sopdet/models-q38/mmproj-Qwen3.8-27B-bf16.gguf --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --chat-template-file /home/sopdet/models-q38/quimmedes-chat-template.jinja \
  --chat-template-kwargs '{"reasoning_effort":"low|medium|high|max"}' \
  --reasoning-budget 2048|4096|8192|16384 \
  --reasoning-budget-message "I am thinking for too long -- let me gather more info about the task." \
  -c 200000
```
- **The MTP trick**: `--spec-type draft-mtp --spec-draft-n-max 4`. The 27B GGUF has the MTP head
  IN-FILE (`blk.64.nextn.*`), so NO separate draft model needed. ~2x decode speed.
- **turbo2 KV**: `--cache-type-v turbo2` (the TurboQuant+ 2-bit V cache) + `q8_0` K. This is what
  makes 200K ctx fit 24GB.
- **quimmedes chat template** (custom, 7-level reasoning dial): file at
  `~/models-q38/quimmedes-chat-template.jinja`. Enables per-level reasoning instructions + string
  tool args + robust content rendering.
- `--reasoning-budget` = token cap on the thinking block. This model is a reasoning model; without
  a budget it can loop / return empty.

### B. Qwen3.6-35B-A3B MoE (fast agentic coding, 128K) — turboquant, no spec
```bash
llama-server-turbo -ngl 99 --device ROCm0 --port ${PORT} \
  --cache-type-k q8_0 --cache-type-v turbo2 --jinja --parallel 1 \
  -m /home/sopdet/models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf -c 131072
```
IQ4_NL UD quant. 35B-A3B MoE. (Fast per-token; good for coding.)

### C. Qwen3.8-Flash-Next (177B-class MoE, 200K) — CAFE build, not turboquant
```bash
llama-server-cafe -ngl 99 --device ROCm0 --port ${PORT} \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1 \
  -m /data/flash-next/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf \
  --mmproj /data/flash-next/mmproj-Qwen3.8-Flash-Next-F16.gguf \
  -c 200000
```
Uses the **quimmedes/cafe-llama.cpp** fork (MoE host-RAM offload flags: `-hmoe`/`-nhmoe`/`-cmoe`).
28 shards. MTP draft for Flash-Next exists separately (`/data/flash-next/mtp/mtp-...Q4_K_M.gguf`).

### D. Cafe 27B (qwen3.8-27b-cafe) — the SAME Qwen3.8-27B file, on the cafe fork, live-effort
```bash
llama-server-cafe -ngl 99 --device ROCm0 --port ${PORT} \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1 --no-mmproj-offload \
  -m /home/sopdet/models-q38/Qwen3.8-27B-Q5-XYZ-v2.gguf \
  --mmproj /home/sopdet/models-q38/mmproj-Qwen3.8-27B-bf16.gguf \
  --chat-template-file quimmedes-chat-template.jinja \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' --reasoning-budget 4096 \
  -c 200000
```
NOTE: cafe build CANNOT do MTP on this model (GGML_ASSERT crash) and `-fa on` is broken for long
ctx on ROCm — plain attention, q8_0/q8_0. Kept for live per-request effort (via :5900 proxy).

### E. Small models — all turboquant, q8_0/q8_0 KV
- `lfm2.5-2.6b`  = LFM2.5-2.6B Q6_K, `-c 131072` (pure-reasoning small agent/tool model)
- `lfm2.5-vl-3b` = LFM2.5-VL-3B Q6_K + mmproj, `-c 131072` (local vision)
- `qwen3-vl-8b`  = Qwen3-VL-8B Q4_K_M + mmproj, `-c 32768` (vision QA)

## 6. PERFORMANCE (measured on this exact 7900 XTX)
- Qwen3.8-27B Q5-XYZ-v2 + MTP(n4) + turbo2 V: ~35 t/s raw, ~70 t/s accepted. 16.7GB VRAM at 200K ctx.
- DeltaNet/hybrid scan costs ~30% vs a plain dense model (DeepSeek-V4-Flash Q4_K_M = 92.8 t/s on same fork).
- Flash-Next (cafe, no FA): 40K-token prompt 60.8s (~659 t/s prefill), 80K in 98.4s (~813 t/s, linear).
- MTP on Qwen3.6-35B-A3B: not used (no in-file head); Flash-Next has a separate MTP draft GGUF.

## 7. PITFALLS / NOTES FOR THE SAME-CARD FRIEND
1. **The ambient LD_LIBRARY_PATH will shadow the ROCm libs** and give `undefined symbol` errors —
   always launch via the env wrapper / source the env first.
2. **turboquant = the MTP machine**; **cafe = the Flash-Next/MoE-offload machine**. Different forks,
   different features. Don't expect turboquant to load Flash-Next or cafe to do MTP on the 27B.
3. **MTP needs `-np 1`** (parallel 1).
4. **Reasoning model**: give `--reasoning-budget` or it loops/returns empty.
5. **VRAM ceiling (24GB)**: 27B @ 200K + mmproj sits ~23.4GB. Loading a 2nd big model swaps it out
   (llama-swap evicts). Fine for one-at-a-time.
6. **System-RAM OOM at deep ctx**: this build keeps context checkpoints in system RAM. At ~154K
   tokens turboquant hit 41GB anon-RSS and the OOM killer took down the desktop. 64GB total RAM is
   the ceiling, not 200K ctx. Watch RSS if you run long dsh sessions.
7. **dsh / agent harness**: llama-swap on :8080 is the single OpenAI-compatible endpoint everything
   points at. Per-effort = per-model-id (qwen38-q5*), no per-request effort on turboquant.
8. `--no-mmproj-offload` keeps the vision projector in system RAM (frees VRAM for ctx); on a 24GB
   card you want that for the 27B.

---

## APPENDIX — llama-swap config.yaml (VERBATIM, as running 2026-09-03)

```yaml
logLevel: debug
logToStdout: both
# llama-swap config - turboquant backend
healthCheckTimeout: 300

models:
  "lfm2.5-2.6b":
    name: "LFM2.5-2.6B Q6_K (Liquid, pure-reasoning agent/tool)"
    aliases:
      - "lfm"
    cmd: >
      /home/sopdet/.local/bin/llama-server-turbo
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1
      -m /home/sopdet/models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q6_K.gguf
      -c 131072

  "lfm2.5-vl-3b":
    name: "LFM2.5-VL-3B Q6_K (Liquid vision)"
    aliases:
      - "lfm-vl"
      - "lfmvl"
    cmd: >
      /home/sopdet/.local/bin/llama-server-turbo
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1
      -m /home/sopdet/models/LFM2.5-VL-3B-GGUF/LFM2.5-VL-3B-Q6_K.gguf
      --mmproj /home/sopdet/models/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-F16.gguf
      -c 131072

  "qwen38-q5":
    name: "Qwen3.8-27B Q5-XYZ-v2 (quimmedes, draft-mtp)"
    aliases:
      - "q38q5"
      - "qwen38"
    cmd: >
      /home/sopdet/.local/bin/llama-server-turbo
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v turbo2 --jinja --parallel 1
      --metrics
      -m /home/sopdet/models-q38/Qwen3.8-27B-Q5-XYZ-v2.gguf
      --mmproj /home/sopdet/models-q38/mmproj-Qwen3.8-27B-bf16.gguf
      --no-mmproj-offload
      --spec-type draft-mtp --spec-draft-n-max 4
      --chat-template-file /home/sopdet/models-q38/quimmedes-chat-template.jinja
      --chat-template-kwargs '{"reasoning_effort":"low"}'
      --reasoning-budget 2048
      --reasoning-budget-message "I am thinking for too long -- let me gather more info about the task."
      -c 200000

  "qwen38-q5-med":
    name: "Qwen3.8-27B Q5 (quimmedes, MEDIUM effort)"
    aliases:
      - "q38med"
    cmd: >
      /home/sopdet/.local/bin/llama-server-turbo
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v turbo2 --jinja --parallel 1
      --metrics
      -m /home/sopdet/models-q38/Qwen3.8-27B-Q5-XYZ-v2.gguf
      --mmproj /home/sopdet/models-q38/mmproj-Qwen3.8-27B-bf16.gguf
      --no-mmproj-offload
      --spec-type draft-mtp --spec-draft-n-max 4
      --chat-template-file /home/sopdet/models-q38/quimmedes-chat-template.jinja
      --chat-template-kwargs '{"reasoning_effort":"medium"}'
      --reasoning-budget 4096
      --reasoning-budget-message "I am thinking for too long -- let me gather more info about the task."
      -c 200000

  "qwen38-q5-hi":
    name: "Qwen3.8-27B Q5 (quimmedes, HIGH effort)"
    aliases:
      - "q38hi"
    cmd: >
      /home/sopdet/.local/bin/llama-server-turbo
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v turbo2 --jinja --parallel 1
      --metrics
      -m /home/sopdet/models-q38/Qwen3.8-27B-Q5-XYZ-v2.gguf
      --mmproj /home/sopdet/models-q38/mmproj-Qwen3.8-27B-bf16.gguf
      --no-mmproj-offload
      --spec-type draft-mtp --spec-draft-n-max 4
      --chat-template-file /home/sopdet/models-q38/quimmedes-chat-template.jinja
      --chat-template-kwargs '{"reasoning_effort":"high"}'
      --reasoning-budget 8192
      --reasoning-budget-message "I am thinking for too long -- let me gather more info about the task."
      -c 200000

  "qwen38-q5-max":
    name: "Qwen3.8-27B Q5 (quimmedes, MAX effort)"
    aliases:
      - "q38max"
    cmd: >
      /home/sopdet/.local/bin/llama-server-turbo
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v turbo2 --jinja --parallel 1
      --metrics
      -m /home/sopdet/models-q38/Qwen3.8-27B-Q5-XYZ-v2.gguf
      --mmproj /home/sopdet/models-q38/mmproj-Qwen3.8-27B-bf16.gguf
      --no-mmproj-offload
      --spec-type draft-mtp --spec-draft-n-max 4
      --chat-template-file /home/sopdet/models-q38/quimmedes-chat-template.jinja
      --chat-template-kwargs '{"reasoning_effort":"max"}'
      --reasoning-budget 16384
      --reasoning-budget-message "I am thinking for too long -- let me gather more info about the task."
      -c 200000

  "qwen3.6-35b-a3b":
    name: "Qwen3.6-35B-A3B MoE (fast agentic coding, 128k)"
    aliases:
      - "qwen"
    cmd: >
      /home/sopdet/.local/bin/llama-server-turbo
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v turbo2 --jinja --parallel 1
      -m /home/sopdet/models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
      -c 131072

  "qwen3-vl-8b":
    name: "Qwen3-VL-8B-Instruct (local vision QA, GPU)"
    aliases:
      - "vl"
      - "vision"
      - "qwen3vl"
    cmd: >
      /home/sopdet/.local/bin/llama-server-turbo
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1
      -m /home/sopdet/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf
      --mmproj /home/sopdet/models/mmproj-Qwen3VL-8B-Instruct-F16.gguf
      -c 32768
    ttl: 600

  "qwen3.8-flash-next":
    name: "Qwen3.8-Flash-Next 125B hybrid (cafe, live effort, vision)"
    aliases:
      - "flash"
      - "flash-next"
      - "q38-flash"
    cmd: >
      /home/sopdet/.local/bin/llama-server-cafe
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      -fa on -nhmoe 36 --ngram-ssd
      --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1
      -m /data/flash-next/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf
      --mmproj /data/flash-next/mmproj-Qwen3.8-Flash-Next-F16.gguf
      --image-min-tokens 1024
      -c 200000
    ttl: 600

  "qwen3.8-27b-cafe":
    name: "Qwen3.8-27B Q5-XYZ (cafe, live effort, vision)"
    aliases:
      - "q38cafe"
      - "qwen38-cafe"
    cmd: >
      /home/sopdet/.local/bin/llama-server-cafe
      -ngl 99 --device ROCm0 --host 127.0.0.1 --port ${PORT}
      --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1
      --no-mmproj-offload
      -m /home/sopdet/models-q38/Qwen3.8-27B-Q5-XYZ-v2.gguf
      --mmproj /home/sopdet/models-q38/mmproj-Qwen3.8-27B-bf16.gguf
      --chat-template-file /home/sopdet/models-q38/quimmedes-chat-template.jinja
      --chat-template-kwargs '{"reasoning_effort":"medium"}'
      --reasoning-budget 4096
      --reasoning-budget-message "I am thinking for too long -- let me gather more info about the task."
      -c 200000
    ttl: 600
```


---

## APPENDIX B - The HipFire foray & DFlash2 eval (same-card experiments)

Context: this all ran on the SAME RX 7900 XTX 24GB. It is what I tested alongside turboquant
when hunting for more speed. TL;DR: **stayed on turboquant + MTP**, but these are the
benchmarks + the two alternative engines in case your friend wants to try them.

### 1. DFlash2 spec-decoding (llama.cpp PR #27342 upstream build, 2026-08-20)
Scratch build: `~/llama-cpp-dflash` (branch `pr-dflash2`, commit `5ecbe1a`, upstream
ggml-org/llama.cpp + DFlash2). Drafters: `~/models-dflash/`.

| method | drafter speed | acceptance | note |
|---|---|---|---|
| AR baseline | - | - | 37.4 tok/s |
| DFlash2 + Q4_K_M drafter | ~60 tok/s | 43% | max ~128K ctx at 90% VRAM (q4_0 KV) |
| DFlash2 + Q8_0 drafter | ~69 tok/s | 49% | fails >64K ctx |
| **MTP (turboquant, n=4)** | ~65 tok/s | **57%** | **keeps full 200K ctx** |

Verdict: DFlash2 **ties MTP on speed** but cannot reach 200K ctx on 24GB (upstream has no
turbo2 KV), so **MTP stays**. PR #27342 still open/unmerged. Model was
`Qwen3.8-27B-Q5-XYZ-v2` on the 7900 XTX. Drafters are the incoai
`Qwen3.8-27B-DFlash2-{Q4_K_M,Q8_0}.gguf` (1.1GB / 2.0GB).

### 2. hipfire (warpfront/hipfire) - RDNA-native Rust engine, INSTALLED + TESTED on this box
**Actually installed and tested (2026-09-01)** - `~/.hipfire`, commit `6163369`, gfx1100.
This is NOT vendor marketing; these are from the local serve.log.

Setup on this box:
- Install: `~/.hipfire/bin/hipfire` (install.json: commit 6163369329d3b076376286a00c13cadbae069ecc, rocm_root /opt/rocm, gfx1100, profile auto)
- Model: `qwen3.8:27b-mq5` (18.7 GB, custom MQ5 quant, 64-layer hybrid - 48 DeltaNet linear + 16 full-attn confirmed in load log)
- DFlash2 draft: `qwen38-27b-dflash-mq4.hfq` (1.2 GB, 5-layer draft, W=2048 window, block=8)
- config.toml: dflash=on, mtp=off, kv_cache=q8, max_seq=131072, kv_adaptive=off
- Daemon served OpenAI-compatible on :11435 (`hipfire serve -d`)
- Prefill: weight sweep ~4.4s, warm prefill; KV Q8 VMM (16/64 layers carry KV, 262144 phys cap)

**Real measured decode (from ~/.hipfire/serve.log, drafter=dflash):**
- Low-tau runs (tau ~2, cold/code): **35-40 tok/s** (e.g. 37.0/35.1/37.9/39.5 @ tau 1.97-2.16)
- High-tau runs (tau 8-13, warm cache, code): **90-164 tok/s** (147.3/159.3/150.1/163.9 @ tau 11-12; 109.2/97.7/90.3/116.1/72.4 @ tau 6.6-12.9)
- Single-token warm spikes: up to **602-640 tok/s** (cache-resumed, 1-tok decode)
- The spread is genre/tau/context dependent - DFlash wins big on warm code with high tau.

**Issues hit (from log):**
- `open think span at end of generation (validation)` - streaming completions FAILED and rolled back on some attempts (attempt 2/3) when the model left a think span open. Thinking-disabled configs dropped reasoning_effort (warning spam).
- `reasoning_effort`/`reasoning.effort` INVALID CONFIG warnings - hipfire drops effort when thinking is disabled.

Verdict: installed + tested, real speedup on warm/high-tau code (up to ~160 tok/s vs ~35 cold).
Still stayed on turboquant + MTP as the daily driver (MTP + turbo2 KV + 200K ctx + the whole
quimmedes tooling is more complete for agent work), but hipfire genuinely runs the 27B faster
in the right conditions.
### 3. ROCm HIP-graph exec-update wedge (rocm-systems#10021 / OhgurLabs fixes)
This one is a **warning for anyone running llama.cpp-ROCm deep-context on 24GB** -
the same workload class as turboquant at 200K ctx.
- Under device-memory exhaustion, `hipGraphExecUpdate` can (1) crash, (2) **silently
  execute stale kernel args while reporting success**, (3) malformed AQL packet -> queue
  abort (one host leaked 25.7GB in KFD, needed reboot).
- **Both PRs (#10022 + #10714) still open/unmerged; defects still ship in ROCm 10.0.0**
  and the 10.1.0 nightly (as of 2026-08-27).
- Fixes repo: OhgurLabs/rocm-hipgraph-fixes (patch + build libamdhip64, use via LD_PRELOAD).
- **Workaround with NO patches:** `GGML_CUDA_DISABLE_GRAPHS=1` (slower, crash-free), or
  the Vulkan backend (never affected).
- KFD teardown leak (bug 3) fixed upstream in Linux 6.15 - this box is on 7.0.0-30, so
  that one is gone; the userspace patches are still needed on every kernel to stop the
  crash/silent-corruption that triggers the fault.
- My 7900 XTX deep runs (up to ~217K ctx on the 27B) are exactly the trigger class, so
  this is on the radar - but not patched in, since production has not hit the VRAM-floor
  crash yet (the 2026-09-02 OOM was system-RAM context checkpoints, a different thing).

### 4. How it all fits (what I actually run)
- **Daily driver:** turboquant (`feature/turboquant-kv-cache`) + MTP + turbo2 KV + llama-swap
  - everything in the main doc.
- **Tested, rejected for prod:** DFlash2 (ties MTP, loses ctx). hipfire: installed + tested
  (real 90-164 tok/s warm/high-tau on the 27B), but turboquant + MTP remains the daily driver
  for the fuller agent tooling; hipfire has no Flash-tier. cafe-llama for the 27B: MTP crashes
  that build; used only for Flash-Next + live-effort.
- **ROCm wedge:** known risk on deep 24GB runs; `GGML_CUDA_DISABLE_GRAPHS=1` is the
  no-build safety valve if it ever bites.
