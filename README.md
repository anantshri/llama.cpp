# fleet-opt-20261110 - combined fleet performance branch

> **Fleet deploy branch.** Not stock llama.cpp and not an upstream PR branch:
> upstream master `45c8ea837` (pinned 2026-11-10) plus 55 cherry-picked
> commits covering the current SYCL / CUDA / Metal performance work, built
> and gated as one tree. This is the branch production serving runs.

## Carry manifest (55 commits)

| PR | What it brings |
|---|---|
| #30606 | SYCL: route BMG D=256 single-query decode to the FA vector kernel |
| #31208 | fattn-mkl host-wait check removed from SYCL-graph compatibility checks |
| #31138 | oneDNN FA cache made thread-safe + runtime-failure memoization |
| #31536 | fattn-mkl fix (2-line subset): mask broadcast over heads/batches |
| #31428 | SYCL: ESIMD-only subset (3 commits) - grouped XMX GEMM without SLM staging (fused/grouped MoE GEMM) |
| #31748 | SYCL: large-register-file flash_attn_tile |
| #31753 | SYCL: fused ADD / RMS_NORM / MUL + vectorized RMS norm |
| #31802 | CUDA: fused q8_0/q4_0 KV dequant inside the MMA flash-attention kernels |
| #31641 | CUDA: FA MMA ncols2 picked with least head padding for large batches (also accuracy-beneficial) |
| #31533 | CUDA: few-row MMQ for BS 2-24 |
| #30489 | CUDA: fuse GLU into the MMQ Q8_1 activation quantization |
| #31850 | llama: upload expert copies on a separate stream, with an event-lifetime fix |
| #30047 | Metal: few-row MMA kernels for MUL_MAT_ID and more src0 types |
| #30728 | Metal: opt-in GQA vector-decode fast path |
| #28439 | Metal: wide query tile for kernel_flash_attn_ext (+ per-device tuning) |
| #25556 | Metal: q8_0 KV optimization |
| #27441 | Metal: skinny-batch mul_mm tiles + deterministic split-K for small-batch decode |
| #25453 | Metal: extend small-batch mat-vec dispatch to 16 rows |
| #31390 | Metal: fuse sigmoid + bias routing in TOPK_MOE |

The upstream base itself already carries the recently merged SYCL work this
model leans on: DPAS flash-attention for quantized KV caches (#31125), SYCL
graph record/replay (#28725), multi-column XMX MMVQ (#29864), vectorized
reordered dequant stores (#30976), MMVQ templating (#31013), DMMV cleanup
(#31325), permuted/batched quantized-KV FA strides (#30761).

Test-only fix on top of the 55: `tests/test-backend-ops.cpp` loop-closer fix
before #25556's inserted test block (branch tree built at `1438246ad` + fix).

Gates: SYCL MUL_MAT_ID+REUSE 1722/1722 (f8_e4m3 excluded - upstream gap:
master's test matrix lists the type, ggml-sycl has no convert for it),
FLASH_ATTN_EXT hsk=256 kv>=1024 10/10, RMS_NORM/ADD/MUL f32 167/167, MUL_MAT
q8_0/q5_K/iq3_s 259/259; CUDA MUL_MAT/MUL_MAT_ID 4113/4113, FLASH_ATTN_EXT
4936/4947 (hsk=256 GQA long-KV error 0.0007-0.0014 vs 5e-4 tol reproduces on
plain master - pre-existing upstream, reduced by #31641).

## Measured impact on qwen3.8-flash-next (MoE, SYCL, Intel Arc Pro B70 32 GB)

vs the previous fleet branch (2026-10-21 pin), production env, llama-bench,
one round per leg, batch shape `-b 1024 -ub 256` on both arms (the `-ub 512`
flag below stacks on top of these):

| leg | prev | this branch | delta |
|---|---:|---:|---:|
| pp512 | 780.3 | 995.4 | +27.6% |
| pp@8k | 746.4 | 967.4 | +29.6% |
| pp@32k | 667.3 | 838.2 | +25.6% |
| pp@64k | 596.9 | 724.5 | +21.4% |
| pp@131k | 489.6 | 569.0 | +16.2% |
| tg128 @ ctx0 | 27.04 | 27.45 | +1.5% |
| tg@8k | 25.36 | 26.35 | +3.9% |
| tg@32k | 22.03 | 25.48 | +15.6% |
| tg@64k | 18.57 | 24.03 | +29.4% |
| tg@131k | 14.15 | 21.73 | +53.5% |

Headline: the deep-context decode collapse is fixed - prev fell -48% (27.0 ->
14.2 tok/s) from ctx0 to 131k, this branch holds -21% (27.5 -> 21.7).
Attribution: the DPAS FA kernel reads q8_0 KV natively (#31125); the previous
build had no DPAS path and this was the deep-decode bottleneck. Prefill
attribution candidates: mmq.cpp rework, ESIMD grouped MoE GEMM (#31428),
reordered-store vectorization (#30976), fused elementwise (#31753 series);
FA-dispatch probes on this build confirm oneDNN SDPA is the optimal prefill
dispatch.

Follow-up env deploy (2026-11-11, on top of the table above): the flash-next
wrapper adds GGML_SYCL_SPARSE_FA=1 + GGML_SYCL_ENABLE_GRAPH=1 - decode FA
gathers only the indexer-selected KV cells: tg131k 21.65 -> 25.17 (+16.3%)
sparse solo, -> 26.09 +/- 0.30 (+20.5%) with graph. Correctness gate: deep
needles at 25/50/90% of a ~106k haystack, 6/6 byte-identical vs dense FA.
Lane scoreboard vs the 2026-10-21 prod baseline: pp131k +37.6%, tg131k
+84% (14.15 -> 26.09 tok/s).

## Flags in use for this model (all measured on the B70)

Serving flags (deployed entry):

| flag | measured impact on flash-next |
|---|---|
| `-b 2048 -ub 512` | +13.0% pp@8k / +18.4% pp@32k / +18.5% pp@131k (derived 674 tok/s), tg flat, 131k VRAM-verified. The single biggest config lever on this branch; halves the oneDNN tiny-GEMM call count. (Previously `-b 1024 -ub 256`.) |
| `-fa on` + `-ctk q8_0 -ctv q8_0` | enables the DPAS FA quantized-KV decode path - the deep-decode fix above (tg@131k 14.2 -> 21.7 tok/s). F16 KV would bypass it. |
| `-ngl 999` | full offload (29.6 GB weights on GPU; the second GGUF shard is a disk-backed n-gram, off-GPU). |
| `-c 131072` | measured q8_0-KV ceiling on this card: loads at 131072, fails at 196608 (q4_0 KV reaches 196608 at lower attention fidelity). |
| `-lm mmap -lzm on` | mmap-backed weight loading, in place for the verified 131k @ ub512 VRAM fit (VRAM-check rc=0); the 28.8 GB shard-2 n-gram stays disk-backed, off-GPU. |
| `-np 1`, `--jinja`, `--reasoning-effort xhigh`, `--top-k 20 --min-p 0.0` | functional (Qwen chat template, xhigh reasoning, sampling profile); no perf delta measured. |

Env knobs (single-knob A/B vs production env; pg legs r=2, tg confirm r=5;
the deployed-knobs row is the TG campaign, r=3, tg128 @ depth 131072):

| env | measured impact |
|---|---|
| `GGML_SYCL_SPARSE_FA=1` + `GGML_SYCL_ENABLE_GRAPH=1` | **deployed 2026-11-11** in the flash-next-only wrapper (`llama-server-flashnext-env`; not fleet-wide): tg128@131k 21.65 -> 25.17 (+16.3%, sparse solo) -> 26.09 +/- 0.30 (+20.5%, both), pp flat; graph solo +3.1% tg@ctx0 (28.14 +/- 0.22 vs 27.30 +/- 0.27) and +3.1% tg@depth (22.33 +/- 0.23 vs 21.65 +/- 0.14, r=3); an earlier r=5 ctx0 confirm read +4.2%. Correctness: deep needles 6/6 byte-identical vs dense FA. Not enabled fleet-wide: sparse-FA unproven for other archs, graph mechanism partially unresolved (possibly its async-USM side effect). |
| `GGML_SYCL_ENABLE_XMX_ESIMD=1`, `GGML_SYCL_XMX_MAX_COLS=8`, `GGML_SYCL_XMX_GLU_MAX_COLS=8`, `GGML_SYCL_USE_LEVEL_ZERO_API=0`, `ONEAPI_DEVICE_SELECTOR=level_zero:0` | all within +/-0.6% of the production env (inside +/-2% run noise) - env is at the plateau; kept for reproducibility. |
| `GGML_SYCL_BMG_VRAM_CHECK` | build-side VRAM-fit gate; rc=0 verified the 131k @ ub512 fit. |

FA dispatch (prefill probes, pp131k): oneDNN SDPA (default) is optimal -
MKL-only -17.7%, TILE-only -71%; forcing DPAS into prefill shapes is 3x
slower (the FA_DPAS_MAX_Q=31 cap is correct policy for prefill).
GGML_SYCL_SPARSE_FA=1 is a prefill no-op (the qsa sparse hint routes decode
FA only) - its win is the decode-side KV gather above.

In flight (side trees, not part of this branch): lightning-indexer
query-tile rewrite - +4.1% pp131k, PPL parity (+0.03%) - gated WIN awaiting
promotion; FA_DPAS_MAX_Q runtime knob - rejected, evidence kept.

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

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

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
| [CANN](docs/build.md#cann) | Huawei Ascend NPU |
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
