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

---

> [!IMPORTANT]
> **Fork: Intel Arc (Xe2 / Battlemage) SYCL performance build.**
> Tag lineage: `v0.4.1-sycl1` (SYCL kernel patches) → `v0.4.1-sycl2` (this
> tag: + oneDNN build enablement & strict CMake gating + this README).
> Based on upstream 0.4.1 (`7cf1c54a9`). Developed and measured on an
> **Intel Arc Pro B70 32 GB** with oneAPI 2026.1 (`icpx`) and oneDNN
> 2026.0.2. Everything below this section is the upstream README, unchanged.

## What is different here

| # | Change | Type | Upstream status |
|---|---|---|---|
| 1 | **MKL-FA softmax: coalesce loads** — one work-group per row instead of one work-item per row; kernel 3.1–3.8× faster; prefill gain grows with context (+7% @9k → +49% @127k) | SYCL kernel (`v0.4.1-sycl1`) | PR #28918, open |
| 2 | **Decode op fusions** — MMVQ GLU fusion extended to mixed quant types; new rms_norm+scale and ssm_conv+silu fusions; decode +1–2% | SYCL kernels (`v0.4.1-sycl1`) | PR #28931, open |
| 3 | **oneDNN (DNNL) SDPA linked in — required for the numbers below** — with oneDNN linked, all flash-attention prefill is served by oneDNN SDPA instead of the hand-rolled MKL-FA pipeline: qwen3.8-27B pp64000 **+18.7%** on top of the kernel patches | Build enablement (`v0.4.1-sycl2`) | n/a (build config) |
| 4 | **Strict oneDNN CMake gating** — auto-detects the oneAPI-bundled DNNL (`DNNLROOT` env or `/opt/intel/oneapi/dnnl/latest`); **fails the configure** if oneDNN was requested but missing, CPU-only (Debian/Ubuntu `libdnnl-dev`), or target-mismatched — instead of silently compiling it out and losing 20–40% prefill | CMake (`v0.4.1-sycl2`) | this fork |

## Build (Intel GPU; tested on Arc Pro B70)

Prerequisites: Intel oneAPI toolkit 2026.x (`icpx`) and, **from the oneAPI
repo — not Debian's `libdnnl-dev`, which is CPU-only and rejected by this
fork**:

```sh
sudo apt install intel-oneapi-dnnl intel-oneapi-dnnl-devel
```

```sh
. /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=icpx \
      -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON \
      -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
      -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
      -DGGML_SYCL_DNN=ON
cmake --build build-sycl --config Release -j4
```

- Configure output **must** contain `Found oneDNN:` — with this fork the
  configure fails loudly otherwise (upstream 0.4.1 silently builds without
  oneDNN and gives up 20–40% long-prompt prefill).
- `-j4` is deliberate: `icpx` 2026.1 reproducibly segfaults compiling
  ggml-sycl's fattn template instances at high parallelism (`-j12` on a
  12-core/125 GB box — compiler-internal, not memory). `-j4` is safe.
- Run-time env: `ONEAPI_DEVICE_SELECTOR=level_zero:0` and
  `GGML_SYCL_USE_LEVEL_ZERO_API=0`.

## Measured performance (Arc Pro B70, llama-bench, r3 medians)

**1. Kernel patches only — `v0.4.1-sycl1` vs upstream stock 0.4.1, each
model at its production flags, pp64000 (tok/s):**

| model | shape | stock | fork | Δ |
|---|---|---:|---:|---:|
| qwen38-27b-obliterated | Q4_K_XL, q8_0 KV, ub2048 | 626 | 896 | **+43.2%** |
| qwen36-27b | MTP, ub2048 | 621 | 887 | **+42.9%** |
| qwythos-9b | Q8, ub2048 | 1990 | 2718 | **+36.5%** |
| qwen38-27b | Q4_K_XL, f16 KV, ub1024 | 587 | 786 | **+33.9%** |
| qwen38-27b-uncensored | self-quant Q4, f16 KV, ub1024 | 584 | 785 | **+34.5%** |
| laguna-s | MoE, ncmoe 46, ub4096 | 289 | 391 | **+35.0%** |
| gemma4-26b | A4B MoE, Q6_K_XL, ub2048 | 1604 | 2078 | **+29.6%** |
| spark25-4b | dense, ub2048 | 1179 | 1487 | **+26.1%** |
| thomson-small | A3B, ub1024 | 1240 | 1529 | **+23.3%** |
| btl4-compact | 2.3 bpw, ub1024 | 982 | 1154 | **+17.6%** |
| ornith15-35b | A3B MoE+GDN, ncmoe 12, ub1024 | 635 | 706 | **+11.1%** |
| glm47-flash | MLA (deepseek2), ub1024 | 432 | 429 | −0.8% (control) |

Decode (tg128/256) is flat by design, −0.1% to +1.8%. 11 of 12 entries take
the patched MKL-FA softmax path; glm47-flash (MLA) is the negative control —
its attention does not route through the changed kernel. Component timers:
softmax −65…−72% in every row; all other components unchanged.

**2. This tag — kernels + oneDNN SDPA (`v0.4.1-sycl2`), same harness:**

| model | state | pp64000 | Δ |
|---|---|---:|---:|
| qwen3.8-27B | fork kernels (ub4096) | 919.0 | |
| qwen3.8-27B | + oneDNN SDPA | **1090.7** | **+18.7%** |
| ornith-1.5-35B | fork kernels (ub2048, ncmoe 12) | 1043.0 | |
| ornith-1.5-35B | + oneDNN SDPA | **1129.1** | **+8.2%** |

Stacked vs upstream stock 0.4.1 (kernels + oneDNN + staged ubatch):
qwen3.8-27B pp64000 587 → **1090.7 (+85.9%)**; ornith-1.5-35B 635 →
**1129.1 (+77.8%)**. Decode ≥0.97× everywhere; generated text byte-identical
to stock in the parity battery.

**3. Serving path — llama-server, qwen3.8-27B, full stack:**

| prompt size | TTFT stock 0.4.1* | TTFT this fork | pp tok/s (this fork) |
|---|---:|---:|---:|
| 804 tok | ≈0.99 s | 0.92 s | 928 |
| 7 k tok | ≈7.4 s | 5.44 s | 1326 |
| 50 k tok | **81 s** | **30.6 s** | 1149 |

\* stock TTFT at 0.8k/7k derived from measured stock pp rates (810/948/622
tok/s); the 81 s at 50k is measured. The stock build's deep-prefill decay
(948 → 622 tok/s from 7k → 50k) is inverted on this fork (1326 → 1149 —
throughput now holds with depth).

## Caveats

- **MLA models (glm47-flash / deepseek2 arch) do not benefit**: oneDNN and
  MKL both structurally decline MLA's DKQ≠DV head shapes (576≠512), so MLA
  prefill stays on the TILE path and decays with context on *any* build —
  route deep prompts to a GQA/hybrid-attention model instead.
- All numbers: single Arc Pro B70, direct llama-bench at production flags,
  r3 medians, F16 KV unless noted. Your mileage will vary with quant, ubatch
  and prompt depth — the gain scales with the softmax/attention share of the
  prefill wall.

---

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
