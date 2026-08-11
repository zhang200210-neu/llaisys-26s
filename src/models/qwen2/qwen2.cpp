#include "qwen2.hpp"

#include "llaisys/ops.h"

#include "../../utils.hpp"
#include "../../core/context/context.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace llaisys::models {


namespace {

// 获取默认设备 ID
inline int defaultDeviceID(const std::vector<int> &device_ids) {
    return device_ids.empty() ? 0 : device_ids[0];
}

// 将设备 tensor 数据拷贝到主机并转换为 float 向量
std::vector<float> deviceToHostFloat(const std::byte *data,
                                     llaisysDataType_t dtype,
                                     size_t vocab) {
    std::vector<float> host(vocab, 0.0f);
    if (dtype == LLAISYS_DTYPE_F32) {
        const auto *vals = reinterpret_cast<const float *>(data);
        std::copy(vals, vals + vocab, host.begin());
    } else if (dtype == LLAISYS_DTYPE_F16) {
        const auto *vals = reinterpret_cast<const fp16_t *>(data);
        for (size_t i = 0; i < vocab; ++i)
            host[i] = utils::cast<float>(vals[i]);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        const auto *vals = reinterpret_cast<const bf16_t *>(data);
        for (size_t i = 0; i < vocab; ++i)
            host[i] = utils::cast<float>(vals[i]);
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    return host;
}

// 从 logits tensor 提取到 CPU 端的 float 向量
std::vector<float> logitsToHost(llaisysTensor_t logits,
                                llaisysDataType_t dtype,
                                llaisysDeviceType_t device,
                                int device_id,
                                size_t vocab) {
    if (device == LLAISYS_DEVICE_CPU) {
        const auto *src = reinterpret_cast<const std::byte *>(tensorGetData(logits));
        return deviceToHostFloat(src, dtype, vocab);
    }

    const size_t bytes = vocab * utils::dsize(dtype);
    std::vector<std::byte> tmp(bytes);
    llaisys::core::context().setDevice(device, device_id);
    llaisys::core::context().runtime().api()->memcpy_sync(
        tmp.data(), tensorGetData(logits), bytes, LLAISYS_MEMCPY_D2H);
    return deviceToHostFloat(tmp.data(), dtype, vocab);
}

// 从 logits tensor 在设备端直接 argmax（支持 CPU / GPU）
int64_t argmaxFromLogits(llaisysTensor_t logits,
                         llaisysDataType_t dtype,
                         llaisysDeviceType_t device,
                         int device_id) {
    size_t one_shape[1] = {1};
    llaisysTensor_t max_idx = tensorCreate(one_shape, 1, LLAISYS_DTYPE_I64, device, device_id);
    llaisysTensor_t max_val = tensorCreate(one_shape, 1, dtype, device, device_id);
    if (!max_idx || !max_val) {
        if (max_idx) tensorDestroy(max_idx);
        if (max_val) tensorDestroy(max_val);
        return -1;
    }

    ::llaisysArgmax(max_idx, max_val, logits);

    int64_t next_token = -1;
    if (tensorGetDeviceType(max_idx) == LLAISYS_DEVICE_CPU) {
        next_token = *reinterpret_cast<const int64_t *>(tensorGetData(max_idx));
    } else {
        llaisys::core::context().setDevice(device, device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(
            &next_token, tensorGetData(max_idx), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    }

    tensorDestroy(max_idx);
    tensorDestroy(max_val);
    return next_token;
}

// 基于采样参数进行 token 采样
int64_t sampleFromLogits(const std::vector<float> &logits,
                         const LlaisysSamplingParams *params) {
    const size_t vocab = logits.size();
    if (vocab == 0) return -1;

    const int top_k = params ? params->top_k : 1;
    const float top_p = params ? params->top_p : 0.0f;
    const float temperature = params ? params->temperature : 0.0f;
    const uint32_t seed = params ? params->seed : 0u;

    // 贪婪解码
    if (temperature <= 0.0f && top_k <= 1 && top_p <= 0.0f) {
        return static_cast<int64_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    std::vector<int> indices(vocab);
    std::iota(indices.begin(), indices.end(), 0);

    // top-k 过滤
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab) {
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        indices.resize(top_k);
    }

    // 温度缩放
    const float temp = temperature > 0.0f ? temperature : 1.0f;
    std::vector<float> scaled;
    scaled.reserve(indices.size());
    for (int idx : indices)
        scaled.push_back(logits[idx] / std::max(temp, 1e-6f));

    // softmax
    const float max_logit = *std::max_element(scaled.begin(), scaled.end());
    std::vector<float> probs(scaled.size());
    float sum = 0.0f;
    for (size_t i = 0; i < scaled.size(); ++i) {
        probs[i] = std::exp(scaled[i] - max_logit);
        sum += probs[i];
    }
    if (sum <= 0.0f) return indices.front();
    for (float &p : probs) p /= sum;

    // top-p 过滤
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<size_t> order(probs.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return probs[a] > probs[b]; });

        float cumulative = 0.0f;
        size_t keep = 0;
        for (size_t idx : order) {
            cumulative += probs[idx];
            ++keep;
            if (cumulative >= top_p) break;
        }

        std::vector<int> new_indices;
        std::vector<float> new_probs;
        new_indices.reserve(keep);
        new_probs.reserve(keep);
        for (size_t i = 0; i < keep; ++i) {
            new_indices.push_back(indices[order[i]]);
            new_probs.push_back(probs[order[i]]);
        }
        indices.swap(new_indices);
        probs.swap(new_probs);

        const float new_sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (new_sum > 0.0f)
            for (float &p : probs) p /= new_sum;
    }

    // 随机采样
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (r <= cumulative) return indices[i];
    }
    return indices.back();
}

// 根据 logits tensor 获取下一个 token（支持 argmax 或采样）
int64_t nextTokenFromLogits(llaisysTensor_t logits,
                            llaisysDataType_t dtype,
                            llaisysDeviceType_t device,
                            int device_id,
                            size_t vocab,
                            const LlaisysSamplingParams *params) {
    if (!params)
        return argmaxFromLogits(logits, dtype, device, device_id);
    auto host_logits = logitsToHost(logits, dtype, device, device_id, vocab);
    return sampleFromLogits(host_logits, params);
}

// 创建形状为 {1, vocab} 的 logits tensor
llaisysTensor_t createLogitsTensor(llaisysDataType_t dtype,
                                   llaisysDeviceType_t device,
                                   int device_id,
                                   size_t vocab) {
    size_t shape[2] = {1, vocab};
    return tensorCreate(shape, 2, dtype, device, device_id);
}

// 创建形状为 {nseq, vocab} 的打包 logits tensor
llaisysTensor_t createPackedLogitsTensor(llaisysDataType_t dtype,
                                         llaisysDeviceType_t device,
                                         int device_id,
                                         size_t nseq,
                                         size_t vocab) {
    size_t shape[2] = {nseq, vocab};
    return tensorCreate(shape, 2, dtype, device, device_id);
}

} // anonymous namespace


Qwen2::Qwen2(const LlaisysQwen2Meta &meta,
             const LlaisysQwen2Weights &weights,
             llaisysDeviceType_t device,
             const std::vector<int> &device_ids)
    : _meta(meta),
      _weights(&weights),
      _device(device),
      _device_ids(device_ids),
      _decoder(transformer::DecoderConfig{
                   meta.dtype,
                   meta.nlayer,
                   meta.hs,
                   meta.nh,
                   meta.nkvh,
                   meta.dh,
                   meta.di,
                   meta.maxseq,
                   meta.voc,
                   meta.epsilon,
                   meta.theta},
               &weights,
               device,
               device_ids) {}

Qwen2::~Qwen2() {
    clearPackedState();
}

void Qwen2::resetKVCache() {
    clearPackedState();
    _decoder.resetKVCache();
}

void Qwen2::setKVCacheEnabled(bool enabled) {
    _decoder.setKVCacheEnabled(enabled);
}

void Qwen2::setTensorParallel(llaisysComm_t comm, llaisysStream_t stream, int tp_size) {
    _decoder.setTensorParallel(comm, stream, tp_size);
}

void Qwen2::setKVContext(void *ctx, size_t past_len_tokens) {
    clearPackedState();
    _kv_ctx = ctx;
    if (ctx)
        _decoder.bindExternalKVContext(ctx, past_len_tokens);
    else
        _decoder.clearExternalKVContext();
}

void *Qwen2::getKVContext() const {
    return _kv_ctx;
}

int Qwen2::exportKVContext(void *ctx, size_t block_tokens) {
    return _decoder.exportKVContext(ctx, block_tokens);
}

void Qwen2::clearPackedState() {
    for (auto *ctx : _packed_kv_contexts) {
        if (ctx) ::llaisysQwen2KVContextRelease(ctx);
    }
    _packed_kv_contexts.clear();
    _packed_prompts.clear();
}


int64_t Qwen2::infer(const int64_t *token_ids, size_t ntoken) {
    return prefill(token_ids, ntoken);
}

int64_t Qwen2::prefill(const int64_t *token_ids, size_t ntoken) {
    if (!token_ids || ntoken == 0) return -1;
    clearPackedState();

    const int device_id = defaultDeviceID(_device_ids);
    llaisysTensor_t logits = createLogitsTensor(_meta.dtype, _device, device_id, _meta.voc);
    if (!logits) return -1;

    if (!_decoder.prefill(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    const int64_t next_token = argmaxFromLogits(logits, _meta.dtype, _device, device_id);
    tensorDestroy(logits);
    return next_token;
}

int64_t Qwen2::step(const int64_t *token_ids, size_t ntoken) {
    if (!token_ids || ntoken == 0) return -1;
    clearPackedState();

    const int device_id = defaultDeviceID(_device_ids);
    llaisysTensor_t logits = createLogitsTensor(_meta.dtype, _device, device_id, _meta.voc);
    if (!logits) return -1;

    if (!_decoder.decodeStep(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    const int64_t next_token = argmaxFromLogits(logits, _meta.dtype, _device, device_id);
    tensorDestroy(logits);
    return next_token;
}

bool Qwen2::prefillPacked(const int64_t *token_ids,
                          size_t ntoken,
                          const int64_t *token_offsets,
                          size_t nseq,
                          int64_t *out_next_tokens) {
    if (!token_ids || !token_offsets || nseq == 0 || ntoken == 0 || !out_next_tokens)
        return false;
    clearPackedState();

    if (token_offsets[0] != 0 || static_cast<size_t>(token_offsets[nseq]) != ntoken)
        return false;
    for (size_t i = 0; i < nseq; ++i) {
        if (token_offsets[i] >= token_offsets[i + 1]) return false;
    }

    const int device_id = defaultDeviceID(_device_ids);
    llaisysTensor_t logits = createPackedLogitsTensor(_meta.dtype, _device, device_id, nseq, _meta.voc);
    if (!logits) return false;

    if (!_decoder.prefillPacked(token_ids, ntoken, token_offsets, nseq, logits)) {
        tensorDestroy(logits);
        return false;
    }

    for (size_t i = 0; i < nseq; ++i) {
        llaisysTensor_t row = tensorSlice(logits, 0, i, i + 1);
        if (!row) {
            tensorDestroy(logits);
            return false;
        }
        out_next_tokens[i] = argmaxFromLogits(row, _meta.dtype, _device, device_id);
        tensorDestroy(row);
    }
    tensorDestroy(logits);

    // 保存每个序列的 prompt token
    _packed_prompts.resize(nseq);
    for (size_t i = 0; i < nseq; ++i) {
        const size_t begin = static_cast<size_t>(token_offsets[i]);
        const size_t end   = static_cast<size_t>(token_offsets[i + 1]);
        _packed_prompts[i].assign(token_ids + begin, token_ids + end);
    }

    // 为每个序列构建独立的 KV 快照
    constexpr size_t kBlockTokens = 64;
    _packed_kv_contexts.assign(nseq, nullptr);

    llaisysTensor_t single_logits = createLogitsTensor(_meta.dtype, _device, device_id, _meta.voc);
    if (!single_logits) {
        clearPackedState();
        return false;
    }

    for (size_t i = 0; i < nseq; ++i) {
        _decoder.resetKVCache();
        _decoder.clearExternalKVContext();

        const auto &prompt = _packed_prompts[i];
        if (prompt.empty()) {
            tensorDestroy(single_logits);
            clearPackedState();
            return false;
        }
        if (!_decoder.prefill(prompt.data(), prompt.size(), single_logits)) {
            tensorDestroy(single_logits);
            clearPackedState();
            return false;
        }

        auto *ctx = ::llaisysQwen2KVContextCreate(
            _meta.dtype, _device, device_id,
            _meta.nlayer, _meta.nh, _meta.nkvh, _meta.dh);
        if (!ctx || _decoder.exportKVContext(ctx, kBlockTokens) != 0) {
            if (ctx) ::llaisysQwen2KVContextRelease(ctx);
            tensorDestroy(single_logits);
            clearPackedState();
            return false;
        }
        _packed_kv_contexts[i] = ctx;
    }

    _decoder.clearExternalKVContext();
    _decoder.resetKVCache();
    tensorDestroy(single_logits);
    return true;
}

bool Qwen2::stepPacked(const int64_t *token_ids,
                       size_t ntoken,
                       const int64_t *token_offsets,
                       size_t nseq,
                       int64_t *out_next_tokens) {
    if (!token_ids || !token_offsets || nseq == 0 || !out_next_tokens)
        return false;
    if (token_offsets[0] != 0 || static_cast<size_t>(token_offsets[nseq]) != ntoken)
        return false;
    for (size_t i = 0; i < nseq; ++i) {
        if (token_offsets[i] >= token_offsets[i + 1]) return false;
    }
    if (_packed_prompts.size() != nseq || _packed_kv_contexts.size() != nseq)
        return false;

    const int device_id = defaultDeviceID(_device_ids);
    constexpr size_t kBlockTokens = 64;

    // 提取每个序列的当前 token（必须为单步）
    std::vector<int64_t> step_tokens(nseq);
    for (size_t i = 0; i < nseq; ++i) {
        const size_t begin = static_cast<size_t>(token_offsets[i]);
        const size_t end   = static_cast<size_t>(token_offsets[i + 1]);
        if (end - begin != 1) return false; // 仅支持单步
        step_tokens[i] = token_ids[begin];
    }

    std::vector<LlaisysQwen2KVContext *> contexts(_packed_kv_contexts.begin(),
                                                  _packed_kv_contexts.end());
    llaisysTensor_t logits = createPackedLogitsTensor(_meta.dtype, _device, device_id, nseq, _meta.voc);
    if (!logits) return false;

    if (!_decoder.decodePacked(step_tokens.data(), nseq, contexts.data(), logits, kBlockTokens)) {
        tensorDestroy(logits);
        clearPackedState();
        return false;
    }

    for (size_t i = 0; i < nseq; ++i) {
        llaisysTensor_t row = tensorSlice(logits, 0, i, i + 1);
        if (!row) {
            tensorDestroy(logits);
            clearPackedState();
            return false;
        }
        out_next_tokens[i] = argmaxFromLogits(row, _meta.dtype, _device, device_id);
        tensorDestroy(row);
        if (out_next_tokens[i] < 0) {
            tensorDestroy(logits);
            clearPackedState();
            return false;
        }
        // 记录已生成的 token（用于后续可能的串联）
        _packed_prompts[i].push_back(step_tokens[i]);
        _packed_prompts[i].push_back(out_next_tokens[i]);
    }

    tensorDestroy(logits);
    return true;
}



int64_t Qwen2::prefillSampling(const int64_t *token_ids, size_t ntoken,
                               const LlaisysSamplingParams *params) {
    if (!token_ids || ntoken == 0) return -1;
    clearPackedState();

    const int device_id = defaultDeviceID(_device_ids);
    llaisysTensor_t logits = createLogitsTensor(_meta.dtype, _device, device_id, _meta.voc);
    if (!logits) return -1;

    if (!_decoder.prefill(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    const int64_t next_token = nextTokenFromLogits(logits, _meta.dtype, _device, device_id,
                                                   _meta.voc, params);
    tensorDestroy(logits);
    return next_token;
}

int64_t Qwen2::stepSampling(const int64_t *token_ids, size_t ntoken,
                            const LlaisysSamplingParams *params) {
    if (!token_ids || ntoken == 0) return -1;
    clearPackedState();

    const int device_id = defaultDeviceID(_device_ids);
    llaisysTensor_t logits = createLogitsTensor(_meta.dtype, _device, device_id, _meta.voc);
    if (!logits) return -1;

    if (!_decoder.decodeStep(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    const int64_t next_token = nextTokenFromLogits(logits, _meta.dtype, _device, device_id,
                                                   _meta.voc, params);
    tensorDestroy(logits);
    return next_token;
}

} // namespace llaisys::models
