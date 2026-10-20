# fleet-stable-20261021 - rolling fleet branch

> **Private fleet deploy branch.** Not stock llama.cpp, not an upstream PR
> branch. Rebase of `fleet-stable-20261020` onto upstream master, plus the
> measured SYCL conversion-widening patches from `extending-30775`.

- **Upstream base**: 84e66d032 (pinned at validation 2026-10-20/21; re-pin at
  execution if master moved)
- **Carries from fleet-stable-20261020**: #30604, #29864, #30728, #27902,
  small-KV prefill fix, grammar recursion limit, qwen4exp kpool fix, #30761,
  #29948 (incl. #30489), #28439 + 69 M4 Max rows. #30430 and #30699 dropped
  (landed upstream).
- **New**: PR #30775 (Q8_0 reordered conversion widening, +2.4-3.6% primed
  prefill), reordered Q4_0 conversion widening (+6.4-7.3%), reordered Q4_K
  vec4 conversion stores (+14.7-18.7% on Q4_K-dominant models). Q5_K left at
  vec2: the vec4 remap measured within noise and is not included.
- Full 17-model sweep and per-patch data: machine-config
  `dumbo/patches/FLEET-STABLE-BRANCH-20261021.md` and
  `bench-results/pr30775-sweep-20261020/`.

## Full-model sweep: this branch vs upstream master (5ba3b9fd3)

Single round per arm (5 reps/test, arms alternated per model), primed protocol
(-fa 1 -p 1,512,2048 -n 128 -r 5 -b 2048 -ub 512 -t 6, full GPU offload, F16 KV),
Intel Arc Pro B70. Primed prefill delta = this branch vs upstream master:

| Model | Quant mix | pp512 | pp2048 | tg128 |
|---|---|---:|---:|---:|
| Qwen3-4B | Q4_0 | +6.39% | +6.44% | -0.04% |
| Qwen3-4B | Q8_0 | +2.35% | +2.46% | -0.01% |
| Spark-X2.5-4B-Uncensored | Q8_0 | +2.45% | +2.60% | -0.02% |
| Spark-X2.5-4B | F16 (unquantized control) | -0.02% | -0.10% | -0.00% |
| Qwythos-9B | Q8_0 | +2.71% | +3.08% | +0.03% |
| BTL-4-Compact | IQ2_XXS (MoE) | +2.15% | +1.91% | +0.00% |
| OrcaCyber-Nano-1.0 | IQ4_XS-heavy MoE mix | -0.01% | -0.02% | +0.28% |
| Qwen3.8-27B | Q4_0 | +7.13% | +7.29% | -0.02% |
| Qwen3.8-27B-Uncensored | Q4_K_M | +18.34% | +18.74% | +0.03% |
| GLM-4.7-Flash | UD-Q4_K_XL (MoE) | +0.91% | +0.81% | -0.03% |
| Qwen3.6-27B-MTP | UD-Q4_K_XL | +14.72% | +15.21% | -0.01% |
| Qwen3.8-27B | UD-Q4_K_XL (q5_K-heavy) | -0.04% | -0.15% | +0.01% |
| Swift-Qwen3.8-27B | Q4_K_M | +14.69% | +15.34% | +0.04% |
| Thomson-1.0-Small | Q4_K_M | +1.71% | +2.04% | +0.20% |
| Qwen3.8-27B-OBLITERATED | Q6_K | +0.30% | +0.11% | +0.01% |
| gemma-4-26B-A4B | UD-Q6_K_XL (MoE) | +0.31% | +0.61% | -0.01% |
| Qwen3.8-27B | Q8_0 | +3.44% | +3.58% | +0.01% |

Pattern: gains track the quant mix. PPL parity on all models. Skipped for
VRAM: Ornith-1.5-35B (37.8 GB), Laguna-S-2.1 (68/96 GB).

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

## Fleet patch: SYCL reordered Q4_0 conversion widening

Widens the reordered-Q4_0 dequant conversion: flat launch, four qs bytes per
work-item, two vector stores; identical math, bit-identical output.
Fleet (Arc Pro B70): +6.4-7.1% primed prefill on Q4_0 models (27B: 886 -> 949
t/s pp512; 4B: 5165 -> 5497 t/s); fresh-layout prefill and decode unchanged;
PPL within run noise; q4_0 MUL_MAT gates 111/111.

## Fleet patch: SYCL reordered Q4_K/Q5_K conversion store widening

Vectorizes the reordered-Q4_K/Q5_K dequant conversion stores (vec4 for Q4_K,
vec2 for Q5_K; same per-element math, bit-identical output).
Fleet (Arc Pro B70): primed prefill uplift on Q4_K/Q5_K models measured on
Qwen3.8-27B-UD-Q4_K_XL and Swift-Qwen3.8-27B-Q4_K_M - numbers recorded in
bench-results/pr30775-q4proto-20261020.
