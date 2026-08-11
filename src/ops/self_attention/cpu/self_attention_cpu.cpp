#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
    template <typename T>
    void self_attn_impl(std::byte *const out,
                        const std::byte *const q,
                        const std::byte *const k,
                        const std::byte *const v,
                        const size_t qlen,
                        const size_t kvlen,
                        const size_t nhead,
                        const size_t nkvh,
                        const size_t dim,
                        const size_t dv,
                        const float scale) {
        const auto *q_ptr = reinterpret_cast<const T *>(q);
        const auto *k_ptr = reinterpret_cast<const T *>(k);
        const auto *v_ptr = reinterpret_cast<const T *>(v);
        auto       *out_ptr = reinterpret_cast<T *>(out);

        const size_t q_head_stride  = dim;
        const size_t k_head_stride  = dim;
        const size_t v_head_stride  = dv;
        const size_t q_seq_stride   = nhead * dim;
        const size_t k_seq_stride   = nkvh * dim;
        const size_t v_seq_stride   = nkvh * dv;
        const size_t out_head_stride = dv;
        const size_t out_seq_stride = nhead * dv;

        const int head_factor = static_cast<int>(nhead / nkvh);
        const int causal_limit = static_cast<int>(qlen - kvlen); // used below: s + kvlen - qlen

        std::vector<float> logits(kvlen);
        std::vector<float> probs(kvlen);

        for (size_t s = 0; s < qlen; ++s) {
            const int allow_upto = static_cast<int>(s) + causal_limit;

            for (size_t h = 0; h < nhead; ++h) {
                const T *q_vec = q_ptr + s * q_seq_stride + h * q_head_stride;
                const int kh = static_cast<int>(h / head_factor);
                const T *k_base = k_ptr + kh * k_head_stride;
                const T *v_base = v_ptr + kh * v_head_stride;

                float max_logit = -std::numeric_limits<float>::infinity();

                // logits computation with causal mask
                for (size_t t = 0; t < kvlen; ++t) {
                    float logit;
                    if (static_cast<int>(t) > allow_upto) {
                        logit = -1e20f;
                    } else {
                        const T *k_vec = k_base + t * k_seq_stride;
                        float dot = 0.0f;
                        for (size_t j = 0; j < dim; ++j) {
                            dot += llaisys::utils::cast<float>(q_vec[j]) *
                                   llaisys::utils::cast<float>(k_vec[j]);
                        }
                        logit = dot * scale;
                    }
                    logits[t] = logit;
                    max_logit = std::max(max_logit, logit);
                }

                // softmax
                float sum_exp = 0.0f;
                for (size_t t = 0; t < kvlen; ++t) {
                    const float e = std::exp(logits[t] - max_logit);
                    probs[t] = e;
                    sum_exp += e;
                }
                const float inv_sum = 1.0f / sum_exp;

                // weighted sum of values
                T *y = out_ptr + s * out_seq_stride + h * out_head_stride;
                for (size_t d = 0; d < dv; ++d) {
                    float acc = 0.0f;
                    for (size_t t = 0; t < kvlen; ++t) {
                        const T *v_vec = v_base + t * v_seq_stride;
                        acc += (probs[t] * inv_sum) * llaisys::utils::cast<float>(v_vec[d]);
                    }
                    y[d] = llaisys::utils::cast<T>(acc);
                }
            }
        }
    }

    template <typename T>
    void self_attn_segmented_impl(std::byte *const out,
                                  const std::byte *const q,
                                  const std::byte *const k,
                                  const std::byte *const v,
                                  const size_t qlen,
                                  const size_t kvlen,
                                  const size_t nhead,
                                  const size_t nkvh,
                                  const size_t dim,
                                  const size_t dv,
                                  const float scale,
                                  const int64_t *const q_offsets,
                                  const int64_t *const kv_offsets,
                                  const size_t nseg) {
        const auto *q_ptr = reinterpret_cast<const T *>(q);
        const auto *k_ptr = reinterpret_cast<const T *>(k);
        const auto *v_ptr = reinterpret_cast<const T *>(v);
        auto       *out_ptr = reinterpret_cast<T *>(out);

        const size_t q_head_stride  = dim;
        const size_t k_head_stride  = dim;
        const size_t v_head_stride  = dv;
        const size_t q_seq_stride   = nhead * dim;
        const size_t k_seq_stride   = nkvh * dim;
        const size_t v_seq_stride   = nkvh * dv;
        const size_t out_head_stride = dv;
        const size_t out_seq_stride = nhead * dv;
        const int head_factor = static_cast<int>(nhead / nkvh);

        std::vector<float> logits(kvlen);
        std::vector<float> probs(kvlen);

        // Build query -> segment mapping
        std::vector<size_t> q2seg(qlen, 0);
        for (size_t seg = 0; seg < nseg; ++seg) {
            const size_t qb = static_cast<size_t>(q_offsets[seg]);
            const size_t qe = static_cast<size_t>(q_offsets[seg + 1]);
            for (size_t s = qb; s < qe; ++s) {
                q2seg[s] = seg;
            }
        }

        for (size_t s = 0; s < qlen; ++s) {
            const size_t seg = q2seg[s];
            const size_t q_begin   = static_cast<size_t>(q_offsets[seg]);
            const size_t q_end     = static_cast<size_t>(q_offsets[seg + 1]);
            const size_t kv_begin  = static_cast<size_t>(kv_offsets[seg]);
            const size_t kv_end    = static_cast<size_t>(kv_offsets[seg + 1]);

            const size_t local_q = s - q_begin;
            const size_t seg_qlen = q_end - q_begin;
            const size_t seg_kvlen = kv_end - kv_begin;
            const size_t local_allow = local_q + (seg_kvlen - seg_qlen);
            const size_t global_allow = kv_begin + local_allow;

            for (size_t h = 0; h < nhead; ++h) {
                const T *q_vec = q_ptr + s * q_seq_stride + h * q_head_stride;
                const int kh = static_cast<int>(h / head_factor);
                const T *k_base = k_ptr + kh * k_head_stride;
                const T *v_base = v_ptr + kh * v_head_stride;

                float max_logit = -std::numeric_limits<float>::infinity();

                for (size_t t = 0; t < kvlen; ++t) {
                    float logit;
                    const bool in_segment = (t >= kv_begin && t < kv_end);
                    const bool causal_ok  = (t <= global_allow);
                    if (!in_segment || !causal_ok) {
                        logit = -1e20f;
                    } else {
                        const T *k_vec = k_base + t * k_seq_stride;
                        float dot = 0.0f;
                        for (size_t j = 0; j < dim; ++j) {
                            dot += llaisys::utils::cast<float>(q_vec[j]) *
                                   llaisys::utils::cast<float>(k_vec[j]);
                        }
                        logit = dot * scale;
                    }
                    logits[t] = logit;
                    max_logit = std::max(max_logit, logit);
                }

                float sum_exp = 0.0f;
                for (size_t t = 0; t < kvlen; ++t) {
                    const float e = std::exp(logits[t] - max_logit);
                    probs[t] = e;
                    sum_exp += e;
                }
                const float inv_sum = 1.0f / sum_exp;

                T *y = out_ptr + s * out_seq_stride + h * out_head_stride;
                for (size_t d = 0; d < dv; ++d) {
                    float acc = 0.0f;
                    for (size_t t = 0; t < kvlen; ++t) {
                        const T *v_vec = v_base + t * v_seq_stride;
                        acc += (probs[t] * inv_sum) * llaisys::utils::cast<float>(v_vec[d]);
                    }
                    y[d] = llaisys::utils::cast<T>(acc);
                }
            }
        }
    }
} // anonymous namespace

namespace llaisys::ops::cpu {
void self_attention(std::byte *const out,
                    const std::byte *const q,
                    const std::byte *const k,
                    const std::byte *const v,
                    const llaisysDataType_t type,
                    const size_t qlen,
                    const size_t kvlen,
                    const size_t nhead,
                    const size_t nkvh,
                    const size_t dim,
                    const size_t dv,
                    const float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attn_impl<float>(out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attn_impl<llaisys::bf16_t>(out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attn_impl<llaisys::fp16_t>(out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

void self_attention_segmented(std::byte *const out,
                              const std::byte *const q,
                              const std::byte *const k,
                              const std::byte *const v,
                              const llaisysDataType_t type,
                              const size_t qlen,
                              const size_t kvlen,
                              const size_t nhead,
                              const size_t nkvh,
                              const size_t dim,
                              const size_t dv,
                              const float scale,
                              const int64_t *const q_offsets,
                              const int64_t *const kv_offsets,
                              const size_t nseg) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attn_segmented_impl<float>(
            out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale, q_offsets, kv_offsets, nseg);
    case LLAISYS_DTYPE_BF16:
        return self_attn_segmented_impl<llaisys::bf16_t>(
            out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale, q_offsets, kv_offsets, nseg);
    case LLAISYS_DTYPE_F16:
        return self_attn_segmented_impl<llaisys::fp16_t>(
            out, q, k, v, qlen, kvlen, nhead, nkvh, dim, dv, scale, q_offsets, kv_offsets, nseg);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
