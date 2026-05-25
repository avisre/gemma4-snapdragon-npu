// gemma4_runner.h — Stateless prefill runner for the 5-part Gemma-4 E2B
// split exported via AI Hub for Hexagon v69.
//
// IMPORTANT: this header was redesigned around the actual ONNX/QNN
// signature produced by the split5 export, which does NOT include any
// KV-cache tensors. Each forward pass is a stateless 32-token prefill:
//   input_ids[1,32]  -> part0 -> (h[1,32,1536],
//                                 per_layer_inputs_passthrough[1,32,35,256],
//                                 attn_scratch_a[1,32,1,256] x2,
//                                 attn_scratch_b[1,32,1,512] x2)
//                    -> part1 -> (h[1,32,1536],
//                                 attn_scratch_c[1,1,32,512],
//                                 attn_scratch_d[1,1,32,512],
//                                 attn_scratch_e[1,1,32,32])
//                    -> part2 -> (h[1,32,1536],
//                                 attn_scratch_f[1,1,32,512] x2,
//                                 attn_scratch_g[1,1,32,32])
//                    -> part3 -> norm_h[1,32,1536]
//                    -> part4 -> logits[1,32,262144]
//
// Autoregressive generation = repeatedly run the full chain on the
// rolling 32-token context, take the logit at the last *real* position
// (i.e. token index = num_real_tokens-1), greedy-sample, append to the
// rolling buffer, and shift if we overflow 32. This is slow but
// architecturally correct for the no-KV split.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// QNN handle aliases (opaque to the header).
typedef void *Qnn_BackendHandle_t;
typedef void *Qnn_ContextHandle_t;
typedef void *Qnn_DeviceHandle_t;
typedef void *Qnn_GraphHandle_t;
typedef void *Qnn_LogHandle_t;

namespace gemma4 {

// ---------------------------------------------------------------------------
// Compile-time model constants.
// ---------------------------------------------------------------------------
constexpr int kNumLayers     = 35;
constexpr int kHiddenDim     = 1536;   // inter-part hidden width
constexpr int kKvHeads       = 8;
constexpr int kHeadDim       = 256;
constexpr int kVocabSize     = 262144;
constexpr int kPrefillSeq    = 32;     // FIXED — graphs are statically S=32

// PLE dims live in ple_preprocess.h (gemma4::kPleNumLayers, kPleDim).

// 5-part split: part i owns decoder layers [PartFirstLayer(i), PartLayerEnd(i)).
// Part 4 owns no decoder layers — it is lm_head + final norm only.
constexpr int kNumParts      = 5;
constexpr int kPartLayerEnd[kNumParts]  = { 10, 20, 28, 35, 35 };
constexpr bool kPartHasLmHead[kNumParts] = { false, false, false, false, true };

inline int PartFirstLayer(int part) {
    return part == 0 ? 0 : kPartLayerEnd[part - 1];
}
inline int PartLayerCount(int part) {
    return kPartLayerEnd[part] - PartFirstLayer(part);
}

// ---------------------------------------------------------------------------
// Sampling configuration.
// ---------------------------------------------------------------------------
struct SamplingParams {
    enum class Mode { kGreedy, kTemperature };
    Mode  mode        = Mode::kGreedy;
    float temperature = 1.0f;
    int   top_k       = 0;
    float top_p       = 1.0f;
    uint32_t seed     = 0xC0DEC0DEu;
};

// ---------------------------------------------------------------------------
// Forward declarations.
// ---------------------------------------------------------------------------
class PLEPreprocessor;

class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int32_t> Encode(const std::string& text,
                                        bool add_bos = true) const = 0;
    virtual std::string Decode(const std::vector<int32_t>& ids) const = 0;
    virtual int32_t EosId() const = 0;
    virtual int32_t PadId() const = 0;
};

// ---------------------------------------------------------------------------
// Runner.
// ---------------------------------------------------------------------------
class Gemma4Runner {
public:
    struct Options {
        std::string  qnn_sdk_root;
        std::vector<std::string> context_binary_paths;  // must be kNumParts
        std::string  backend_lib  = "libQnnHtp.so";
        std::string  system_lib   = "libQnnSystem.so";
        int          max_new_tokens     = 32;
        SamplingParams sampling;
    };

    Gemma4Runner();
    ~Gemma4Runner();
    Gemma4Runner(const Gemma4Runner&)            = delete;
    Gemma4Runner& operator=(const Gemma4Runner&) = delete;

    bool Initialize(const Options&   opts,
                    ITokenizer*      tokenizer,
                    PLEPreprocessor* ple);

    // Convenience wrappers.
    std::string Generate(const std::string& prompt);

    // Lower-level API used by the runner binary (avoids tokenizer dep).
    // - context_ids: real prompt token ids (length L <= kPrefillSeq).
    // - Calls the on-token callback for each newly generated token id.
    // Returns the list of generated token ids (excludes the prompt).
    using TokenSink = std::function<void(int32_t, int)>;
    std::vector<int32_t> GenerateFromIds(const std::vector<int32_t>& context_ids,
                                         int max_new_tokens,
                                         const TokenSink& on_token);

    void ResetState();

private:
    bool LoadBackend();
    bool LoadSystem();
    bool CreateDevice();
    // Parse metadata only (cheap), filling part_info_[i].
    bool ParseBinaryMetadata(int part_idx, const std::string& path);
    // Heavy weight load + graph retrieval. Free with FreeContext.
    bool CreateContextFromBinary(int part_idx);
    bool RetrieveGraphHandlesForPart(int part_idx);
    void FreeContext(int part_idx);

    // Run one full chain forward pass over `seq_ids` (length == kPrefillSeq,
    // pad with kPadId on the right). Returns the raw logits buffer for the
    // entire window (shape [kPrefillSeq, kVocabSize], fp16).
    const uint16_t* RunChainOnce(const int32_t* seq_ids);

    int32_t SampleAtPosition(const uint16_t* logits_window, int pos);

private:
    Options          opts_{};
    ITokenizer*      tokenizer_ = nullptr;   // not owned
    PLEPreprocessor* ple_       = nullptr;   // not owned

    // QNN function tables.
    void* backend_dl_ = nullptr;
    void* system_dl_  = nullptr;
    struct QnnInterfaceFns;
    struct QnnSystemInterfaceFns;
    std::unique_ptr<QnnInterfaceFns>       qnn_fns_;
    std::unique_ptr<QnnSystemInterfaceFns> qsys_fns_;

    Qnn_LogHandle_t      log_     = nullptr;
    Qnn_BackendHandle_t  backend_ = nullptr;
    Qnn_DeviceHandle_t   device_  = nullptr;

    std::vector<Qnn_ContextHandle_t> contexts_;   // kNumParts
    // Each part has exactly one graph (no separate prefill/decode).
    std::vector<Qnn_GraphHandle_t>   graphs_;     // kNumParts
    // Cached graph metadata + IO tensor names parsed from the binary.
    struct PartInfo;
    std::vector<std::unique_ptr<PartInfo>> part_info_;  // kNumParts
    // Per-part on-disk blob (the .bin path); loaded lazily during execute.
    std::vector<std::string> part_blob_paths_;

    // Host backing buffers for inter-part data flow (sized for prefill).
    // We allocate one shared set, sized to the max each tensor needs.
    std::vector<uint8_t> buf_input_ids_;     // 32 * sizeof(int32) = 128
    std::vector<uint8_t> buf_per_layer_in_;  // 32 * 35 * 256 * 2 = 573_440
    std::vector<uint8_t> buf_h0_;            // 32 * 1536 * 2     = 98_304
    std::vector<uint8_t> buf_h1_;
    std::vector<uint8_t> buf_h2_;
    std::vector<uint8_t> buf_h3_;
    std::vector<uint8_t> buf_logits_;        // 32 * 262144 * 2   = 16_777_216

    // Attention scratch buffers (sized as in the ONNX inspection).
    // Names are kept short — see RunChainOnce for the mapping.
    std::vector<uint8_t> buf_attn_a0_;  // (1,32,1,256) f16 = 16_384
    std::vector<uint8_t> buf_attn_a1_;
    std::vector<uint8_t> buf_attn_b0_;  // (1,32,1,512) f16 = 32_768
    std::vector<uint8_t> buf_attn_b1_;
    std::vector<uint8_t> buf_attn_c0_;  // (1,1,32,512) f16 = 32_768
    std::vector<uint8_t> buf_attn_c1_;
    std::vector<uint8_t> buf_attn_c2_;  // (1,1,32,32)  f16 = 2_048
    std::vector<uint8_t> buf_attn_d0_;
    std::vector<uint8_t> buf_attn_d1_;
    std::vector<uint8_t> buf_attn_d2_;
};

}  // namespace gemma4
