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

## Fork integration: B70 SYCL, Intel Vulkan, and shared fixes

`sycl-b70-combined-20261001` retains upstream `ggml-org/llama.cpp` master
through its October 1 merge and carries the changes below. The upstream merge
is a snapshot, not a claim that this fork will automatically track later
upstream commits. Only the SYCL and Intel Vulkan changes target the Arc Pro
B70; CUDA and Metal use their own backends from the same source tree.

| Source | Code on this branch (inherited or fork-only) | Reason for inclusion |
| --- | --- | --- |
| [#28918](https://github.com/ggml-org/llama.cpp/pull/28918) | Coalesce oneMKL flash-attention softmax loads on SYCL rather than assigning one work-item per row. | Reduce B70 attention costs; already merged into upstream master before this fork's October merge. |
| [#28931](https://github.com/ggml-org/llama.cpp/pull/28931) | Extend MMVQ GLU fusion to mixed quants and fuse RMS norm with scale and SSM_CONV with SILU on SYCL. | Cut decoder kernel launches; already merged into upstream master before this fork's October merge. |
| [#29107](https://github.com/ggml-org/llama.cpp/pull/29107) | Reorder IQ3_S/IQ3_XXS weights and use reordered SYCL mat-vec and dequant paths. | Improve B70 IQ3 decode; the combined branch carried this before the October merge. |
| [#29171](https://github.com/ggml-org/llama.cpp/pull/29171) | Add SYCL oneMKL flash-attention dispatch for GLM MLA. The separate F16 QK-score storage commit is **reverted**: MKL scores remain F32. | Enable GLM MLA without the demonstrated F16 score-overflow case. |
| [#29186](https://github.com/ggml-org/llama.cpp/pull/29186) | Add reordered Q8_0 wide-load MMVQ and ESIMD DMMV paths in `ggml-sycl`. | On B70, measured Q8_0 decode gains on Qwythos; other weight types are not the target. |
| [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) | Add grouped MoE XMX dequant-GEMM for selected IQ weight types, plus later per-device matrix-shape and type-selection updates. | Group narrow expert GEMMs instead of launching one library GEMM per expert. B70 GSQ-RCO Coder pp8192 rose from about 462 to 764 tok/s with `GGML_SYCL_XMX_GATHER_TYPES=73` (IQ4_NL + IQ3_XXS + IQ2_S). The path uses F16 expert math; this is not a quality-parity claim. |
| [#29375](https://github.com/ggml-org/llama.cpp/pull/29375) | Add multi-column Q5_K SYCL MMVQ handling in `mmvq.cpp` and `vecdotq.hpp`. | Reduce the measured Q5_K n=4 kernel latency; server-side MTP gains remain unverified. |
| [#29608](https://github.com/ggml-org/llama.cpp/pull/29608) | Stage model-weight uploads through a reusable pinned-memory ring. | Reduce warm-cache model load wall time; it does not increase steady-state prompt throughput. |
| [#29338](https://github.com/ggml-org/llama.cpp/pull/29338) | Guard DMMV reads beyond the row tail and add reuse cases to `test-backend-ops`. | Correct an out-of-bounds read on short rows. |
| [#29357](https://github.com/ggml-org/llama.cpp/pull/29357) | Add Intel Vulkan flash-attention prefill shader and dispatch. | Improve B70 Vulkan prefill by about 25% at pp8192 in the measured models; near-full-context Qwen decode timed out and is not validated. |
| [#29751](https://github.com/ggml-org/llama.cpp/pull/29751) | Route Qwen4Exp QSA through the shared indexer k-pool, including sequence-order pools and image-position handling. | Correct Qwen3.8 Flash-Next pooled-attention semantics on the newer upstream base. On B70, a single paired GSQ-RCO Coder tg32 check at depth 32768 improved 20.45 to 22.09 tok/s; short-corpus perplexity differences were within control drift. |
| [#29755](https://github.com/ggml-org/llama.cpp/pull/29755) | Bound grammar parser nesting at 256 levels and test the failure path. | Reject deeply nested client grammars instead of exhausting the server stack; parser test passed on the B70 host. |

#28918 and #28931 were inherited from upstream master, not cherry-picked
into the October fork; the other rows are the fork's selected deltas.

These changes are not a blanket production recommendation. On SYCL,
`GGML_SYCL_XMX_GATHER_TYPES=0` disables the precision-changing grouped path
for a control or quality-sensitive deployment. The F32 QK-score reversal is
independent of that flag. Validate model output and memory headroom before
changing a serving binary; in particular, the combined upstream and PR tree
has not been qualified on CUDA or Metal by these B70 benchmarks.

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
