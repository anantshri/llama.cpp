# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

# fleet-opt-20261127 — combined fleet performance branch

> **Fleet deploy branch.** Not stock llama.cpp: upstream master `15fcfc735`
> (pinned 2026-11-27) plus cherry-picked commits covering the current
> SYCL / CUDA / Metal performance work, built and gated as one tree. This is
> the branch production serving runs.

## Carry manifest (14 PRs)

| PR | What it brings |
| --- | --- |
| #31748 | SYCL: large-register-file flash_attn_tile |
| #32486 | SYCL: env-gated F16-KV decode arm + re-tiled (KG=4) Q8_0 decode arm for fattn-dpas on Xe2 |
| #32944 | SYCL: GDN replay-over-restore port (upstream #31669 shipped without the SYCL part — without this, GDN replay offloads to CPU on SYCL) |
| #32961 | SYCL: MMQ re-enabled for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, with the q8_1 scratch padding zeroed (fixes the bug that got MMQ disabled) |
| #33131 | SYCL: fixed dp4a/XMX MMQ kernels, K-quant (Q2_K–Q6_K) MMQ enablement, reordered-weight MMQ whitelist + GGML_SYCL_FORCE_MMQ runtime env (reconciled with #32961: linear-type tile tuning stays #32961's) |
| #33154 | SYCL: ESIMD DMMV for reordered IQ3_S single-column decode (+12% tg128 on Qwen3.8-27B Q3_K_P) |
| #31993 | SYCL: oneDNN reads reordered Q4_0 as u4 — removes the per-call weight-to-f16 walk (env GGML_SYCL_Q4_0_U4_GEMM to revert) |
| #31802 | CUDA: fused q8_0/q4_0 KV dequant inside the MMA flash-attention kernels (fleet squash of the PR net diff — its head carried 8 upstream merge commits; its kv_rows vec-gather variant was dropped in favor of master's merged use_kv_rows (#32520); the fused-KV MMA routing was kept) |
| #31641 | CUDA: FA MMA ncols2 picked with the least head padding for large batches (also accuracy-beneficial); test matrix keeps master's MLA low-ratio cases and adds the PR's GQA-ratio 5/6 cases |
| #33161 | CUDA: constant V load length when head dim is unpadded (applies to head-dim-256 models) — integrated into the fused-V staging branch from #31802 |
| #31850 | llama: upload expert copies on a separate stream, with an event-lifetime fix |
| #33064 | CUDA: dp4a MMQ compiler-hint fix for CUDA 13 |
| #30728 | Metal: opt-in GQA vector-decode fast path (v3: group split, multi-token verify batches, f16-KV-only restriction) |
| #25453 | Metal: extend small-batch mat-vec dispatch to 16 rows (fleet squash; master's Q2_0 mat-vec type kept) |

Fleet patch on top: lightning-indexer query-tile rewrite (+4.7% pp131k on the
2026-11-10 fleet branch, PPL parity +0.03%, deployed in production since
2026-11-11). The old test-file brace fix is not needed this cycle (its target,
#25556's inserted test block, is not carried).

Dropped from the previous manifest (fleet-opt-20261110) — merged upstream
since the 2026-11-10 pin: #30606, #31208, #31138, #31536, #31428, #31753,
#31533. The base also now carries #30156, #31574, #31669 (GDN replay,
CUDA-side), #31355 and #32520 (kv_rows vec kernel) — #32944 above completes
the GDN replay SYCL side.

## In screening (not carried — blocked or superseded upstream)

- #30489 (CUDA GLU-into-MMQ): collides with merged few-row MMQ (#31533) and
  the MMQ fusion-args work; needs the author's rebase.
- #30047, #28439, #25556, #27441, #31390 (Metal): all stale against the Metal
  backend overhaul (ops split, MUL_MAT_ID map0/amax, TOPK_MOE gating, FA
  value_type refactor); several patch the removed monolithic ggml-metal.metal.
  ⚠ Metal regression risk vs the 2026-11-10 branch (which carried #30047,
  #28439, #25556, #27441, #31390) — the workstation A/B quantifies what the
  refactored layout absorbed.
- #31379 (SYCL GDN+GET_ROWS fusion): blocked upstream on a rebase after #31355.
- #32058 (B70 nsm fix + level-zero leak stopgap), #31234 vs #30792 (CUDA
  native quantized-KV tile kernel), #32626 (ncols2 unification, overlaps
  carried #31641), #33019, #32921, #30645 (Metal tensor-API FA GQA, overlaps
  carried #30728).

