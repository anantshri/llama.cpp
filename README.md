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

## Fork integration: fleet-stable branch

`fleet-stable-20261012` retains upstream `ggml-org/llama.cpp` master
through its October 13 merge (`1b70350c9`, 32 commits, none touching
`ggml/src/ggml-sycl`) and carries the deltas below. The upstream merges
are snapshots, not a claim that this fork will automatically track later
upstream commits. Only the SYCL changes target the Arc Pro B70 host;
CUDA and Metal use their own backends from the same source tree.

| Source | Code on this branch (inherited or fork-only) | Reason for inclusion |
| --- | --- | --- |
| [#29107](https://github.com/ggml-org/llama.cpp/pull/29107) | Reorder IQ3_S/IQ3_XXS weights and use reordered SYCL mat-vec and dequant paths (merged from the PR branch). | B70 IQ3 decode, measured +58 to +82 percent on IQ3 models against master; validated on the B70 host. |
| [#29696](https://github.com/ggml-org/llama.cpp/pull/29696) | Wide vector stores and 256-thread workgroups in the SYCL q4_K/q5_K weight-dequant kernels (merged from the PR head). | Cuts dequant staging cost on K-quant weights; validated on the B70 host in this branch's serving measurements. |
| [#29864](https://github.com/ggml-org/llama.cpp/pull/29864) | Multi-column XMX mat-vec for reordered K-quant and Q8_0 weights (merged from the PR head). | Speculative-decode verification batches; measured +9 to +13 percent decode on Qwen3.8-27B on the B70 host. Deployment note: the branch serves with `GGML_SYCL_XMX_MAX_COLS=8` and `GGML_SYCL_XMX_GLU_MAX_COLS=8`, because the unrestricted wide-batch selection can admit shapes the kernel then declines (review finding); keep those caps until the PR hardens the eligibility gate. |
| [#27902](https://github.com/ggml-org/llama.cpp/pull/27902) | Fix Blackwell IQ quant failures in `ggml-cuda/vecdotq.cuh` (cherry-pick `d7749c08c`; closed unmerged upstream - the defect is a CUDA 13.2 codegen bug, not a code bug). | Corrects IQ decode on the sm120 CUDA host; pending validation on that host. |
| [#27140](https://github.com/ggml-org/llama.cpp/pull/27140) | Fix slow CUDA prefill when the KV cache uses small quantized types (cherry-pick `351d5d982`). | Targets q8_0-and-smaller KV configurations; pending validation on the sm120 CUDA host. |
| [#29755](https://github.com/ggml-org/llama.cpp/pull/29755) | Bound grammar parser nesting at 256 levels and test the failure path (cherry-pick `b61c56fef`). | Rejects deeply nested client grammars instead of exhausting the server stack; validated in production serving on the B70 host. |
| qwen4exp kpool allocation (13-line patch, held in the machine-config repo) | Expand all eight `build_inp_kpool` graph inputs so the allocator always assigns them a buffer. | Robustness for MTP-attached loads with variant draft heads; validated in production serving on the B70 host. |
| [#30374](https://github.com/ggml-org/llama.cpp/pull/30374) | Coalesced dequant of reordered Q4_0/Q8_0 weights for prompt processing (cherry-picks of the PR head `069347263`; supersedes the half-precision helper variant with float math). | Restores post-decode prompt throughput on Q4_0/Q8_0 models; earlier head measured +75 percent pp512 after first decode on a B70 Q8_0 model; the float-math rework of the current head is pending revalidation on the B70 host. |

Inherited via the October 13 upstream merge (not fork deltas):
#30397 (Qwen3.8 Flash-Next MTP image embeddings) and #30331 (laguna
YaRN beta metadata), among the 32 commits. The qwen4exp kpool patch
above applies cleanly on top of the merged tree (verified).

Watchlist (open, not carried): #30430 (get_rows on reordered weights -
correctness fix adjacent to the reorder paths this branch uses; trigger
is rare with llama.cpp's own tied-embedding handling), #30363 (CONV_2D
non-contiguous input fix), #30239 (ESIMD FFN restore, draft).

These changes are not a blanket production recommendation. On SYCL,
`GGML_SYCL_XMX_GATHER_TYPES=0` disables the precision-changing grouped
path for a control or quality-sensitive deployment. Validate model output
and memory headroom before changing a serving binary; the combined tree
is qualified on the Arc Pro B70 host by the measurements above and has
not been qualified on CUDA or Metal by them.

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
