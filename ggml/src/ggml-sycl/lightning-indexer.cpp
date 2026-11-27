#include "lightning-indexer.hpp"
#include "dequantize.hpp"

// Lane slice constants shared by the helper and the kernel (the kernel keeps
// its own constexpr copies with the same values).
static constexpr int64_t LI_LANES = 32;
static constexpr int64_t LI_ELEMS_PER_LANE = 8;

// Load this lane's slice of one K row (one kv cell) into registers.
// Factored out of the kernel so the query-tiled layout can reload the slice
// when a query tile straddles a stream boundary (K slice depends on
// (i_kv, i_stream)). Values loaded per lane are identical to the original
// one-row-per-subgroup kernel.
static inline void load_k_slice(const char * k_base, int64_t lane, ggml_type k_type,
                                float (&k_local)[LI_ELEMS_PER_LANE]) {
    const int64_t lane_base = lane * LI_ELEMS_PER_LANE;
    switch (k_type) {
        case GGML_TYPE_F16: {
            const sycl::half * k_row = (const sycl::half *) k_base;
#pragma unroll
            for (int64_t j = 0; j < LI_ELEMS_PER_LANE; ++j) {
                k_local[j] = static_cast<float>(k_row[lane*LI_ELEMS_PER_LANE + j]);
            }
        } break;
        case GGML_TYPE_F32: {
            const float * k_row = (const float *) k_base;
#pragma unroll
            for (int64_t j = 0; j < LI_ELEMS_PER_LANE; ++j) {
                k_local[j] = k_row[lane*LI_ELEMS_PER_LANE + j];
            }
        } break;
        case GGML_TYPE_BF16: {
            const sycl::ext::oneapi::bfloat16 * k_row = (const sycl::ext::oneapi::bfloat16 *) k_base;
#pragma unroll
            for (int64_t j = 0; j < LI_ELEMS_PER_LANE; ++j) {
                k_local[j] = static_cast<float>(k_row[lane_base + j]);
            }
        } break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1: {
#pragma unroll
            for (int64_t j = 0; j < LI_ELEMS_PER_LANE; ++j) {
                const int64_t idx = lane_base + j;
                const int64_t ib  = idx / QK4_0;
                const int iqs     = idx % (QK4_0/2);
                dfloat2 kv;
                if (k_type == GGML_TYPE_Q4_0) {
                    dequantize_q4_0(k_base, ib, iqs, kv);
                } else if (k_type == GGML_TYPE_Q4_1) {
                    dequantize_q4_1(k_base, ib, iqs, kv);
                } else if (k_type == GGML_TYPE_Q5_0) {
                    dequantize_q5_0(k_base, ib, iqs, kv);
                } else {
                    dequantize_q5_1(k_base, ib, iqs, kv);
                }
                k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<float>(kv.x()) : static_cast<float>(kv.y());
            }
        } break;
        case GGML_TYPE_Q8_0: {
#pragma unroll
            for (int64_t pair = 0; pair < LI_ELEMS_PER_LANE / 2; ++pair) {
                const int64_t elem0 = lane_base + 2 * pair;
                dfloat2 kv;
                dequantize_q8_0(k_base, elem0 / QK8_0, elem0 % QK8_0, kv);
                k_local[2 * pair + 0] = static_cast<float>(kv.x());
                k_local[2 * pair + 1] = static_cast<float>(kv.y());
            }
        } break;
        case GGML_TYPE_IQ4_NL: {
#pragma unroll
            for (int64_t pair = 0; pair < LI_ELEMS_PER_LANE / 2; ++pair) {
                const int64_t elem0 = lane_base + 2 * pair;
                dfloat2 kv;
                dequantize_iq4_nl(k_base, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                k_local[2 * pair + 0] = static_cast<float>(kv.x());
                k_local[2 * pair + 1] = static_cast<float>(kv.y());
            }
        } break;
        default:
#pragma unroll
            for (int64_t j = 0; j < LI_ELEMS_PER_LANE; ++j) {
                k_local[j] = 0.0f;
            }
            break;
    }
}

static void lightning_indexer_f32_sycl(
        const char * q, const char * k, const char * w, const char * m, float * dst,
        int64_t n_embd, int64_t n_head, int64_t n_batch, int64_t n_stream, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        queue_ptr stream) {

    constexpr int64_t LANES = WARP_SIZE;
    constexpr int64_t ELEMS_PER_LANE = 8;
    // Query-tiled layout: one kv cell per WORKGROUP; its SUBGROUPS_PER_BLOCK
    // subgroups walk 16 consecutive query rows each (64 queries per workgroup)
    // so the K slice for that cell is read from DRAM once and the redundant
    // per-subgroup loads hit L1. The original one-row-per-subgroup layout
    // re-read K once per query token (rows = n_batch * n_kv), which dominated
    // this kernel at prefill depths (K DRAM traffic per call was
    // rows * n_embd * sizeof(k elem) - about 8.6 GiB per call at pp131k).
    // Per-row arithmetic is unchanged: same lane/element split, same
    // sequential f32 dot accumulation order, same 32-lane subgroup reduce,
    // same head loop, same score and store formulas. Outputs are not
    // bit-identical in practice (compiler contraction differs at the new
    // loop site, and the downstream top-k selection order is already
    // schedule-dependent run-to-run even on an unchanged binary); the
    // measured effect is PPL parity: 6.1643 vs 6.1628/6.1624 on the same
    // 120k-token wiki slice (dumbo B70, 2026-11-11).
    constexpr int64_t QS_PER_SUBGROUP = 16;
    constexpr int64_t SUBGROUPS_PER_BLOCK = 4;
    constexpr int64_t QUERIES_PER_BLOCK = SUBGROUPS_PER_BLOCK * QS_PER_SUBGROUP;
    constexpr int64_t BLOCK_SIZE = SUBGROUPS_PER_BLOCK * LANES;

    const int64_t n_bs      = n_batch * n_stream;
    const int64_t n_qtiles  = (n_bs + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;
    const int64_t n_groups  = n_kv * n_qtiles;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_groups * BLOCK_SIZE),
            sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const sycl::sub_group sgr = item.get_sub_group();
            const int64_t lane = sgr.get_local_id()[0];
            const int64_t sg   = sgr.get_group_id()[0];

            // one kv cell per work-group; consecutive work-groups cover
            // consecutive kv cells of the same query tile (same store
            // locality as before)
            const int64_t gid    = item.get_group_linear_id();
            const int64_t i_kv   = gid % n_kv;
            const int64_t qtile  = gid / n_kv;
            int64_t      i_bs    = qtile * QUERIES_PER_BLOCK + sg * QS_PER_SUBGROUP;

            // load K row slice into registers (row is contiguous, nbk0 == type
            // size); reused by every query row of the workgroup (reload only on
            // stream change). Guarded: fully out-of-range query tiles do no work.
            if (i_bs < n_bs) {
                int64_t i_stream = i_bs % n_stream;
                const char * k_base = k + i_kv*nbk2 + i_stream*nbk3;
                float k_local[ELEMS_PER_LANE];
                load_k_slice(k_base, lane, k_type, k_local);

            for (int64_t b = 0; b < QS_PER_SUBGROUP && i_bs < n_bs; ++b, ++i_bs) {
                const int64_t cur_batch  = i_bs / n_stream;
                const int64_t cur_stream = i_bs % n_stream;
                if (cur_stream != i_stream) {
                    // query tile straddled a stream boundary: the same kv cell
                    // has a different K slice, reload it
                    k_base = k + i_kv*nbk2 + cur_stream*nbk3;
                    i_stream = cur_stream;
                    load_k_slice(k_base, lane, k_type, k_local);
                }

                const char  * q_base = q + cur_batch*nbq2 + cur_stream*nbq3;
                const float * w_base = (const float *) (w + cur_batch*nbw1 + cur_stream*nbw3);

                float score = 0.0f;
                for (int64_t h = 0; h < n_head; ++h) {
                    const float * q_row = (const float *) (q_base + h*nbq1);
                    float dot = 0.0f;
#pragma unroll
                    for (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                        const int64_t i = lane*ELEMS_PER_LANE + j;
                        if (i < n_embd) {
                            dot += q_row[i] * k_local[j];
                        }
                    }
                    dot = sycl::reduce_over_group(item.get_sub_group(), dot, sycl::plus<float>());
                    if (lane == 0) {
                        score += sycl::max(dot, 0.0f) * w_base[h];
                    }
                }

                if (lane == 0) {
                    const sycl::half * m_base = (const sycl::half *) (m + cur_batch*nbm1 + (cur_stream % nem3)*nbm3);
                    // flat-index store: storing through a strided base pointer
                    // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                    const int64_t dst_idx = i_kv + cur_batch*(nb1/sizeof(float)) + cur_stream*(nb3/sizeof(float));
                    dst[dst_idx] = score + static_cast<float>(m_base[i_kv]);
                }
                }
            }
        });
}

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q8_0 || k->type == GGML_TYPE_Q5_1 || k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q4_1 || k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_IQ4_NL);

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, k, nb);
    GGML_TENSOR_LOCALS(size_t,  nbw, w, nb);
    GGML_TENSOR_LOCALS(int64_t, nem, m, ne);
    GGML_TENSOR_LOCALS(size_t,  nbm, m, nb);
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    // input rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));
    GGML_ASSERT(nb0  == ggml_type_size(dst->type));

    const int64_t n_embd   = neq0;
    const int64_t n_head   = neq1;
    const int64_t n_batch  = neq2;
    const int64_t n_stream = neq3;
    const int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_SIZE * 8);

    lightning_indexer_f32_sycl(
            (const char *) q->data, (const char *) k->data,
            (const char *) w->data, (const char *) m->data, (float *) dst->data,
            n_embd, n_head, n_batch, n_stream, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
