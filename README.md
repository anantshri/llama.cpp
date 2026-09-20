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

## Intel Arc Pro B70 SYCL optimization branch

> This branch is an experimental, continuously maintained Intel Arc Pro B70
> optimization stack based on upstream `ggml-org/llama.cpp`. It is not an
> official upstream release branch.

The `sycl-b70-optimization-anantshri` branch combines the SYCL improvements
submitted upstream by [Anant Shrivastava](https://github.com/anantshri). It
tracks upstream `master` while retaining validated B70 optimizations until they
are merged or superseded upstream.

### Submitted upstream work

| Pull request | Purpose |
|---|---|
| [#28918](https://github.com/ggml-org/llama.cpp/pull/28918) | Coalesce MKL flash-attention softmax loads and replace the serial per-row reduction with a work-group reduction. |
| [#28931](https://github.com/ggml-org/llama.cpp/pull/28931) | Reduce token-generation dispatch overhead with mixed-quant MMVQ GLU and RMS norm/scale fusions. The original SSM fusion was removed after an upstream implementation superseded it. |
| [#29107](https://github.com/ggml-org/llama.cpp/pull/29107) | Add persistent reordered layouts for IQ3_S and IQ3_XXS MMVQ, reorder-aware prefill dequantization, focused backend tests, and oneDNN build detection hardening. |
| [#29171](https://github.com/ggml-org/llama.cpp/pull/29171) | Route GLM-4.7 Flash's 576/512, GQA-20 MLA prompt shape through SYCL MKL flash attention instead of the generic TILE fallback. |

### Measured impact

Benchmarks were run on an Intel Arc Pro B70 with IntelLLVM 2026.1, oneDNN,
F16 KV cache, `-ngl 999 -b 4096 -ub 1024 -fa on`, and three repetitions.
Generation tests produced 256 tokens after loading the stated KV-cache depth.

The table compares clean master with the complete optimization stack.

| Workload | Master | Optimized | Change |
|---|---:|---:|---:|
| Qwen3.8 27B Q4, tg256 at 512 depth | 23.521 tok/s | 23.821 tok/s | +1.28% |
| Qwen3.8 27B Q4, tg256 at 8,192 depth | 22.156 tok/s | 22.458 tok/s | +1.36% |
| Qwen3.8 27B Q4, tg256 at 64,000 depth | 15.122 tok/s | 15.462 tok/s | +2.25% |
| Qwen3.8 27B IQ3_S, tg256 at 512 depth | 12.755 tok/s | 22.991 tok/s | +80.25% |
| Qwen3.8 27B IQ3_S, tg256 at 8,192 depth | 12.356 tok/s | 21.706 tok/s | +75.67% |
| Qwen3.8 27B IQ3_S, tg256 at 64,000 depth | 9.761 tok/s | 14.915 tok/s | +52.80% |
| GLM-4.7 Flash, pp8192 | 429.28 tok/s | 1,586.76 tok/s | +269.63% |
| GLM-4.7 Flash, pp64000 | 67.071 tok/s | 609.595 tok/s | +808.89% |

GLM context-loaded generation remains within 0.38% of master. The GLM change
targets prompt processing and removes the poorly scaling generic attention
fallback rather than changing the decode path.

The standalone GLM MLA change in
[#29171](https://github.com/ggml-org/llama.cpp/pull/29171) improves pp8192 by
201.35% and pp64000 by 463.80%. Combining it with the coalesced MKL softmax
from [#28918](https://github.com/ggml-org/llama.cpp/pull/28918) adds another
22.66% and 61.21%, respectively.

### Attention path notes

oneDNN SDPA remains the preferred and fastest path for supported Qwen
attention shapes. PR #28918 optimizes the MKL flash-attention fallback used
when oneDNN declines a shape, when llama.cpp is built without oneDNN, or when
the GLM MLA optimization explicitly routes its validated shape through MKL.

With oneDNN disabled using `GGML_SYCL_FA_ONEDNN=0`, the optimized MKL path
produced:

| Workload | Master MKL-FA | Optimized MKL-FA | Change |
|---|---:|---:|---:|
| Qwen3.8 Q4, pp8192 | 1,051.12 tok/s | 1,123.18 tok/s | +6.86% |
| Qwen3.8 Q4, pp64000 | 598.01 tok/s | 803.79 tok/s | +34.41% |
| Qwen3.8 IQ3_S, pp8192 | 763.49 tok/s | 800.35 tok/s | +4.83% |
| Qwen3.8 IQ3_S, pp64000 | 490.41 tok/s | 620.82 tok/s | +26.59% |

### Validation

The combined branch passed the targeted changed-path checks:

- IQ2_XS, IQ3_XXS, and IQ3_S `MUL_MAT`: 42/42 tests.
- IQ3 reorder-to-prefill dependency tests: 2/2 tests.
- Exact GLM 576/512 GQA-20 flash-attention case: 1/1 test.
- GLM debug dispatch confirmed MKL with `D=576`, `DV=512`, 20 query heads,
  and one KV head.
- Release IntelLLVM 2026.1 SYCL builds completed with oneDNN, F16, SYCL
  graphs, and Level Zero support.

### Branch maintenance

This branch follows upstream through explicit merge commits. Routine updates
should not rebase or force-push it. When an optimization is accepted or
superseded upstream, the next synchronization should remove the duplicate
implementation and rerun the correctness and performance checks.


## Quick start

A few options to get `llama.cpp` installed on your machine:

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
