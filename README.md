# fleet-stable-20261019 — private fleet deploy branch

> **This is NOT a stock llama.cpp checkout and NOT a PR branch.** It is a
> purpose-built "frankenstein" deploy branch for a private 3-host inference
> fleet (CUDA sm_120 / SYCL Arc B70 / Metal M4 Max): one upstream master pin
> plus validated open PRs plus our own fixes, assembled 2026-10-19.
> Nothing here is ever PR'd upstream — upstream work happens on clean
> single-topic branches. Upstream's README continues below the divider.

- **Base pin**: upstream master `ff5a23376725e5a889fa0bfb987b7541dc304da0` (2026-10-18)
- **Authoritative runbook** (assembly recipe, per-host gates, replay
  commands, rollback): the `machine-config` repo,
  `dumbo/patches/FLEET-STABLE-BRANCH-20261019.md` — this README is a
  manifest, the runbook is the source of truth.
- **Replay gate**: after any rebuild of this branch,
  `git rev-parse HEAD^{tree}` must equal the stage tree below. If it
  doesn't, stop — something diverged.

## Carry manifest (in assembly order)

| # | source | what it buys |
|---|---|---|
| 1 | PR #30430 @ `5c8d9faee` | fix get_rows on reordered quantized weights (NaN on tied-embedding Q8_0; 7/7 types failed on master) |
| 2 | PR #30604 (ours) @ `be1626136` | SYCL IQ3 fused-GEMM decode fix — resolves #30430 conflicts by hand (exported patch in machine-config) |
| 3 | PR #29864 @ `9ac7826ee` | SYCL XMX multicol MMVQ rework (+fused gate/up; cap8 XMX exports) |
| 4 | PR #30728 (ours) @ `d8cf48afc` | Metal GQA vec decode fast path (env-gated `GGML_METAL_FA_VEC_GQA`; +22–48% decode @100k) |
| 5–7 | cherry-picks `d7749c08c` `351d5d982` `b61c56fef` (ours, ex `sycl-b70-combined-20261001`) | sm_120 packed-IQ load fix (#27902, load-bearing), slow-prefill-on-small-KV-quants (#27140), grammar recursion limit (#29495) |
| 8 | cherry-pick PR #30573 @ `9ece86a6c7` | CUDA FA-vec WAR hazard (`__syncwarp`, silent-corruption class) |
| 9 | cherry-pick PR #30699 @ `24be75ecc` | Metal `mul_mm_id_map0<9>` — Bolt (9 experts/token) segfault fix |
| 10 | fleet patch (`patches-upstream/0001-qwen4exp-…kpool.patch`) | qwen4exp MTP kpool abort fix — drop when upstream #30592 merges |
| 11 | PR #30761 @ `eb41b38e7` (Append 1) | SYCL FA permuted/batched quantized-KV dequant — was silently wrong at 2 legal shapes on the prior binary |
| 12 | PR #29948 @ `b6e0f0334` incl. #30489 (Append 1) | CUDA MMQ GLU fusion (+1.31% pp16384 @ prod ubatch on dense Q8_0) |
| 13 | PR #28439 @ `b2d0d2e2b` + 69 M4 Max rows (Append 2) | Metal wide prefill tile, per-device tuned; conflicts with base resolved by hand; rows from the PR's own tuner |

Skipped after measurement (do not re-add without new data): #30775 (flat on
our SYCL stack), #30776 (flat +0.06%), #30226/#30156/#30047/#30645/#30680
(see runbook audit table).

## Stage identity

| stage | HEAD | tree sha (replay gate) |
|---|---|---|
| core (2026-10-19, deployed dumbo+bigdumbo) | `ac38f7c20ec5303a655b229351ed7bfce450cebf` | `f1b9413e6c1b797990b13973f47ca19bc9d9ff7f` |
| + Append 1 | `c55d540f770a0f67e691c0797f817dbe4c859ecc` | `6b49dbdfd17ef196ce17caf645e48bf265bbb5b6` |
| + Append 2 (Metal, pre-README) | `182304f45…` | `38ccc77a3b3e174cae3586c6e82ea3f6e60bb6ec` |

Branch tip identity (including this README commit) lives in the runbook
§Identity — a file cannot embed its own commit's hash.

## Ground rules

- Every carry was gated on the host it affects before append (correctness
  sweeps + PPL parity + perf A/B); gate data lives in the machine-config
  repo (`logs/2026-10-19-phase1b-openpr-validation.md` + per-host logs).
- The two hand-resolved merges (#30604×#30430, #28439×base) are exported as
  patches in `machine-config/dumbo/patches/` — apply those rather than
  re-deriving.
- Rollback on any fleet host = repoint the served binary symlink to the
  prior tree (per-host paths in the runbook).

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

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
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
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
