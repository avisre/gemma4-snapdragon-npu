// gemma4_runner.cpp — Stateless 5-part Gemma-4 E2B prefill driver for v69.
//
// Architecture: see gemma4_runner.h. We parse each .bin's graph signature
// via QnnSystemInterface, allocate host buffers per output tensor, and at
// execute time bind input tensors by NAME (matching the prior part's
// output names — they're stable because the export preserved the ONNX
// names through to the QNN context binary).

#include "gemma4_runner.h"
#include "ple_preprocess.h"

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnLog.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"

#include <dlfcn.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <algorithm>

#define LOGI(fmt, ...) fprintf(stderr, "[gemma4] " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, "[gemma4][E] " fmt "\n", ##__VA_ARGS__)
#define RETURN_IF(cond, msg) do { if (cond) { LOGE("%s", msg); return false; } } while (0)

namespace gemma4 {

// ===========================================================================
// QnnInterface holder
// ===========================================================================
struct Gemma4Runner::QnnInterfaceFns {
    QNN_INTERFACE_VER_TYPE v;
};
struct Gemma4Runner::QnnSystemInterfaceFns {
    QNN_SYSTEM_INTERFACE_VER_TYPE v;
};

// ===========================================================================
// Per-part graph metadata.
//
// We keep a vector of Qnn_Tensor_t input/output descriptors that we update
// per-execute with .v2.clientBuf.{data,dataSize}. The descriptor identity
// (id, name, dims, type) we keep from the binary-info parse.
// ===========================================================================
struct TensorDesc {
    std::string name;
    Qnn_Tensor_t tmpl{};            // template w/ id, name, dims, datatype
    std::vector<uint32_t> dims;     // owning storage
    size_t elem_bytes = 0;
    size_t total_elems = 0;
    void*  host_buf = nullptr;      // borrowed (points into part_info host_storage)
};

struct Gemma4Runner::PartInfo {
    int part_idx = -1;
    std::string graph_name;
    std::vector<TensorDesc> inputs;
    std::vector<TensorDesc> outputs;
    // Owning storage for host buffers, one per output tensor + one per input
    // tensor (input buffers refer to either upstream outputs or newly allocated
    // buffers — we allocate one per input here for safety).
    std::map<std::string, std::vector<uint8_t>> storage;
};

// ===========================================================================
// Lifecycle
// ===========================================================================
Gemma4Runner::Gemma4Runner()
  : qnn_fns_(std::make_unique<QnnInterfaceFns>()),
    qsys_fns_(std::make_unique<QnnSystemInterfaceFns>()) {
    contexts_.assign(kNumParts, nullptr);
    graphs_.assign(kNumParts, nullptr);
    part_info_.resize(kNumParts);
    for (int i = 0; i < kNumParts; ++i) {
        part_info_[i] = std::make_unique<PartInfo>();
        part_info_[i]->part_idx = i;
    }
}

Gemma4Runner::~Gemma4Runner() {
    for (int i = 0; i < kNumParts; ++i) {
        if (contexts_[i] && qnn_fns_->v.contextFree)
            qnn_fns_->v.contextFree(contexts_[i], nullptr);
    }
    if (device_  && qnn_fns_->v.deviceFree)  qnn_fns_->v.deviceFree(device_);
    if (backend_ && qnn_fns_->v.backendFree) qnn_fns_->v.backendFree(backend_);
    if (log_     && qnn_fns_->v.logFree)     qnn_fns_->v.logFree(log_);
    if (backend_dl_) dlclose(backend_dl_);
    if (system_dl_)  dlclose(system_dl_);
}

bool Gemma4Runner::Initialize(const Options&   opts,
                              ITokenizer*      tok,
                              PLEPreprocessor* ple) {
    opts_      = opts;
    tokenizer_ = tok;
    ple_       = ple;
    RETURN_IF(!ple, "null PLE preprocessor");
    RETURN_IF(static_cast<int>(opts_.context_binary_paths.size()) != kNumParts,
              "expected kNumParts context binary paths");

    if (!LoadBackend()) return false;
    if (!LoadSystem())  return false;
    if (!CreateDevice()) return false;

    // Each part's full QNN context is ~1-1.5 GB of weights on HTP. The
    // Hexagon PD has a fixed memory budget that cannot hold all five at
    // once (~5 GB total). We therefore parse the binary metadata up-front
    // (cheap — just tensor descriptors) and LAZILY create the context
    // during execute, freeing it before moving on to the next part.
    //
    // ParseBinaryMetadata fills part_info_[i] without calling
    // contextCreateFromBinary; CreateContextFromBinary in the execute
    // loop does the heavy lift one part at a time.
    for (int i = 0; i < kNumParts; ++i) {
        if (!ParseBinaryMetadata(i, opts_.context_binary_paths[i]))
            return false;
    }

    if (!BuildRopeTensors()) return false;
    return true;
}

// ===========================================================================
// RoPE injection
// ===========================================================================
// AI Hub's compiler dead-code-eliminated the rotary_emb Unsqueeze tensors from
// part 0 (since part 0 owns layers 0..9 but does not itself need to apply
// rotary embedding to its own outputs — those tensors are *intermediate*
// values consumed by self-attention nodes that live in parts 1..3 for layers
// 10..27, which is wrong; layer 0's RoPE is needed by layer 0 inside part 0).
// Empirically, however, parts 1..3 still list the layer-0 sliding and layer-4
// global rotary tensors as graph inputs. They were never produced upstream,
// so the previous runner zero-filled them, corrupting attention.
//
// Here we compute the correct cos/sin tables on the host once, in fp16, and
// later RunChainOnce binds them by name to any part that needs them.
//
// Math (must match transformers/models/gemma4/modeling_gemma4.py exactly):
//   - sliding (layer 0): default RoPE, theta=10000, head_dim=256
//       inv_freq[i] = 1 / 10000^(2i/256), i=0..127  (length 128)
//       freqs[p,i]  = p * inv_freq[i],              shape [32,128]
//       emb         = concat(freqs, freqs, axis=-1) shape [32,256]
//       cos,sin     = cos(emb), sin(emb)            shape [32,256]
//       Layout in QNN buffer: [1,32,256] fp16 row-major = 8192 elems.
//
//   - global (layer 4): proportional RoPE, theta=1e6, head_dim=512,
//                       partial_rotary_factor=0.25
//       rope_angles  = int(0.25 * 512 / 2) = 64
//       inv_freq_rot[i] = 1 / 1e6^(2i/512), i=0..63 (length 64)
//       nope_angles  = 256 - 64 = 192 zeros appended
//       inv_freq     = concat(inv_freq_rot, zeros(192)) length 256
//       freqs[p,i]   = p * inv_freq[i]              shape [32,256]
//       emb          = concat(freqs, freqs, -1)     shape [32,512]
//       cos,sin                                     shape [32,512]
//       Layout in QNN buffer: [1,32,512] fp16 = 16384 elems.
//
// attention_scaling = 1.0 in both cases (verified from the ONNX Constant_4).
// ===========================================================================
static inline uint16_t FloatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t frac = x & 0x7FFFFFu;
    if (exp <= 0) {
        // subnormal or zero
        if (exp < -10) return (uint16_t)sign;
        frac = (frac | 0x800000u) >> (1 - exp);
        // round to nearest
        if (frac & 0x1000u) frac += 0x2000u;
        return (uint16_t)(sign | (frac >> 13));
    } else if (exp >= 31) {
        // inf or nan
        return (uint16_t)(sign | 0x7C00u | (frac ? 0x200u : 0u));
    }
    // round to nearest
    if (frac & 0x1000u) {
        frac += 0x2000u;
        if (frac & 0x800000u) { frac = 0; exp += 1; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (frac >> 13));
}

bool Gemma4Runner::BuildRopeTensors() {
    auto build = [&](const std::string& name_cos,
                     const std::string& name_sin,
                     double base,
                     int head_dim,
                     double partial_rotary_factor) -> void {
        const int seq = kPrefillSeq;
        const int rope_angles = (int)(partial_rotary_factor * head_dim / 2.0);  // 128 sliding, 64 global
        const int half_dim    = head_dim / 2;                                   // 128, 256
        const int full_dim    = head_dim;                                       // 256, 512

        std::vector<double> inv_freq(half_dim, 0.0);
        for (int i = 0; i < rope_angles; ++i) {
            // exponent uses head_dim, not 2*rope_angles (per modeling_rope_utils.py:238)
            inv_freq[i] = 1.0 / std::pow(base, (double)(2 * i) / (double)head_dim);
        }
        // remaining (half_dim - rope_angles) entries stay 0 (nope_angles).

        std::vector<uint8_t>& cos_buf = rope_buffers_[name_cos];
        std::vector<uint8_t>& sin_buf = rope_buffers_[name_sin];
        cos_buf.assign((size_t)seq * full_dim * 2, 0);
        sin_buf.assign((size_t)seq * full_dim * 2, 0);
        auto* cos_p = reinterpret_cast<uint16_t*>(cos_buf.data());
        auto* sin_p = reinterpret_cast<uint16_t*>(sin_buf.data());

        for (int p = 0; p < seq; ++p) {
            for (int i = 0; i < half_dim; ++i) {
                const double angle = (double)p * inv_freq[i];
                const float c = (float)std::cos(angle);
                const float s = (float)std::sin(angle);
                // emb = concat(freqs, freqs, -1) -> position i and i+half_dim
                // share the same angle.
                const size_t off0 = (size_t)p * full_dim + i;
                const size_t off1 = (size_t)p * full_dim + (i + half_dim);
                cos_p[off0] = FloatToHalf(c);
                cos_p[off1] = FloatToHalf(c);
                sin_p[off0] = FloatToHalf(s);
                sin_p[off1] = FloatToHalf(s);
            }
        }
        LOGI("RoPE built: '%s' (cos, %zu bytes), '%s' (sin, %zu bytes), "
             "base=%g head_dim=%d partial=%.2f",
             name_cos.c_str(), cos_buf.size(),
             name_sin.c_str(), sin_buf.size(),
             base, head_dim, partial_rotary_factor);
    };

    // Layer 0 (sliding_attention): default RoPE
    build("_text_model_layers_0_self_attn_Unsqueeze_output_0",     // cos
          "_text_model_layers_0_self_attn_Unsqueeze_1_output_0",   // sin
          /*base*/10000.0, /*head_dim*/256, /*partial*/1.0);

    // Layer 4 (full_attention): proportional RoPE
    build("_text_model_layers_4_self_attn_Unsqueeze_output_0",     // cos
          "_text_model_layers_4_self_attn_Unsqueeze_1_output_0",   // sin
          /*base*/1000000.0, /*head_dim*/512, /*partial*/0.25);

    // Allocate the causal-mask buffer (filled per-RunChainOnce because it
    // depends on num_real_tokens / padding). Key matches the QNN input name
    // that part 2 (and via forwarding, part 3) expect.
    rope_buffers_["_text_model_layers_19_self_attn_Cast_9_output_0"]
        .assign((size_t)kPrefillSeq * kPrefillSeq * 2, 0);
    return true;
}

// Compute [seq, seq] fp16 causal+padding mask:
//   mask[q,k] = 0.0   if k <= q AND input_ids[k] != PAD(0)
//             = -65504.0  otherwise (fp16 min, matches the ONNX Constant_11)
static void FillCausalMaskFp16(uint16_t* dst,
                               const int32_t* seq_ids,
                               int seq) {
    constexpr uint16_t kAllowed = 0x0000u;   // +0.0 fp16
    constexpr uint16_t kMasked  = 0xFBFFu;   // -65504.0 fp16 (largest negative normal)
    for (int q = 0; q < seq; ++q) {
        for (int k = 0; k < seq; ++k) {
            const bool causal_ok = (k <= q);
            const bool key_real  = (seq_ids[k] != 0);
            dst[q * seq + k] = (causal_ok && key_real) ? kAllowed : kMasked;
        }
    }
}

void Gemma4Runner::ResetState() {}

// ===========================================================================
// QNN bring-up
// ===========================================================================
bool Gemma4Runner::LoadBackend() {
    backend_dl_ = dlopen(opts_.backend_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!backend_dl_) {
        // Fall back to full SDK path.
        std::string p = opts_.qnn_sdk_root + "/lib/aarch64-android/" + opts_.backend_lib;
        backend_dl_ = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    }
    RETURN_IF(!backend_dl_, dlerror());

    using GetProvidersFn = Qnn_ErrorHandle_t (*)(
        const QnnInterface_t***, uint32_t*);
    auto get = reinterpret_cast<GetProvidersFn>(
        dlsym(backend_dl_, "QnnInterface_getProviders"));
    RETURN_IF(!get, "QnnInterface_getProviders missing");

    const QnnInterface_t** providers = nullptr;
    uint32_t n = 0;
    RETURN_IF(get(&providers, &n) != QNN_SUCCESS || n == 0, "no QNN providers");
    qnn_fns_->v = providers[0]->QNN_INTERFACE_VER_NAME;

    RETURN_IF(qnn_fns_->v.logCreate(nullptr, QNN_LOG_LEVEL_WARN, &log_) != QNN_SUCCESS,
              "logCreate failed");
    RETURN_IF(qnn_fns_->v.backendCreate(log_, nullptr, &backend_) != QNN_SUCCESS,
              "backendCreate failed");
    return true;
}

bool Gemma4Runner::LoadSystem() {
    system_dl_ = dlopen(opts_.system_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!system_dl_) {
        std::string p = opts_.qnn_sdk_root + "/lib/aarch64-android/" + opts_.system_lib;
        system_dl_ = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    }
    RETURN_IF(!system_dl_, dlerror());

    using GetSysFn = Qnn_ErrorHandle_t (*)(
        const QnnSystemInterface_t***, uint32_t*);
    auto get = reinterpret_cast<GetSysFn>(
        dlsym(system_dl_, "QnnSystemInterface_getProviders"));
    RETURN_IF(!get, "QnnSystemInterface_getProviders missing");

    const QnnSystemInterface_t** sp = nullptr;
    uint32_t n = 0;
    RETURN_IF(get(&sp, &n) != QNN_SUCCESS || n == 0, "no system providers");
    qsys_fns_->v = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    return true;
}

bool Gemma4Runner::CreateDevice() {
    QnnHtpDevice_CustomConfig_t arch_cfg{};
    arch_cfg.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
    arch_cfg.arch.arch = QNN_HTP_DEVICE_ARCH_V69;
    arch_cfg.arch.deviceId = 0;
    QnnDevice_Config_t devCfg{};
    devCfg.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    devCfg.customConfig = &arch_cfg;
    const QnnDevice_Config_t* devCfgs[] = { &devCfg, nullptr };
    RETURN_IF(qnn_fns_->v.deviceCreate(log_, devCfgs, &device_) != QNN_SUCCESS,
              "deviceCreate failed");
    return true;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static size_t DataTypeSize(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_FLOAT_32:
            return 4;
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_FLOAT_16:
            return 2;
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
            return 1;
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
            return 8;
        default:
            return 0;
    }
}

static void FillDesc(TensorDesc& td, const Qnn_Tensor_t& src) {
    td.tmpl = src;  // shallow copy of version + union
    if (src.version == QNN_TENSOR_VERSION_1) {
        const Qnn_TensorV1_t& v1 = src.v1;
        td.name = v1.name ? std::string(v1.name) : std::string();
        td.dims.assign(v1.dimensions, v1.dimensions + v1.rank);
        td.tmpl.v1.dimensions = td.dims.data();
        td.total_elems = 1;
        for (uint32_t d : td.dims) td.total_elems *= d;
        td.elem_bytes  = DataTypeSize(v1.dataType);
    } else {
        const Qnn_TensorV2_t& v2 = src.v2;
        td.name = v2.name ? std::string(v2.name) : std::string();
        td.dims.assign(v2.dimensions, v2.dimensions + v2.rank);
        td.tmpl.v2.dimensions = td.dims.data();
        td.total_elems = 1;
        for (uint32_t d : td.dims) td.total_elems *= d;
        td.elem_bytes  = DataTypeSize(v2.dataType);
    }
}

// Set this tensor's client buffer (handles both V1 and V2).
static void SetClientBuf(Qnn_Tensor_t& t, void* data, size_t bytes) {
    if (t.version == QNN_TENSOR_VERSION_1) {
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf.data     = data;
        t.v1.clientBuf.dataSize = (uint32_t)bytes;
    } else {
        t.v2.memType = QNN_TENSORMEMTYPE_RAW;
        t.v2.clientBuf.data     = data;
        t.v2.clientBuf.dataSize = (uint32_t)bytes;
    }
}

bool Gemma4Runner::ParseBinaryMetadata(int part_idx,
                                       const std::string& path) {
    LOGI("part %d: parsing metadata from %s", part_idx, path.c_str());
    part_blob_paths_.resize(kNumParts);
    part_blob_paths_[part_idx] = path;

    FILE* f = fopen(path.c_str(), "rb");
    RETURN_IF(!f, "cannot open context binary");
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(sz);
    RETURN_IF(fread(blob.data(), 1, sz, f) != (size_t)sz, "read failed");
    fclose(f);

    // Parse binary info to enumerate graphs + tensors.
    QnnSystemContext_Handle_t sysCtx = nullptr;
    RETURN_IF(qsys_fns_->v.systemContextCreate(&sysCtx) != QNN_SUCCESS,
              "systemContextCreate failed");
    const QnnSystemContext_BinaryInfo_t* binInfo = nullptr;
    Qnn_ContextBinarySize_t binSize = 0;
    RETURN_IF(qsys_fns_->v.systemContextGetBinaryInfo(
                  sysCtx, blob.data(), blob.size(), &binInfo, &binSize) != QNN_SUCCESS,
              "systemContextGetBinaryInfo failed");

    // Walk graphs[]. We expect exactly one graph per .bin.
    auto* pi = part_info_[part_idx].get();
    auto extract_graph = [&](const QnnSystemContext_GraphInfoV1_t& g) {
        pi->graph_name = g.graphName ? g.graphName : "";
        pi->inputs.resize(g.numGraphInputs);
        pi->outputs.resize(g.numGraphOutputs);
        for (uint32_t i = 0; i < g.numGraphInputs; ++i) {
            FillDesc(pi->inputs[i], g.graphInputs[i]);
        }
        for (uint32_t i = 0; i < g.numGraphOutputs; ++i) {
            FillDesc(pi->outputs[i], g.graphOutputs[i]);
        }
    };
    auto extract_graphV2 = [&](const QnnSystemContext_GraphInfoV2_t& g) {
        pi->graph_name = g.graphName ? g.graphName : "";
        pi->inputs.resize(g.numGraphInputs);
        pi->outputs.resize(g.numGraphOutputs);
        for (uint32_t i = 0; i < g.numGraphInputs; ++i)
            FillDesc(pi->inputs[i], g.graphInputs[i]);
        for (uint32_t i = 0; i < g.numGraphOutputs; ++i)
            FillDesc(pi->outputs[i], g.graphOutputs[i]);
    };
    auto extract_graphV3 = [&](const QnnSystemContext_GraphInfoV3_t& g) {
        pi->graph_name = g.graphName ? g.graphName : "";
        pi->inputs.resize(g.numGraphInputs);
        pi->outputs.resize(g.numGraphOutputs);
        for (uint32_t i = 0; i < g.numGraphInputs; ++i)
            FillDesc(pi->inputs[i], g.graphInputs[i]);
        for (uint32_t i = 0; i < g.numGraphOutputs; ++i)
            FillDesc(pi->outputs[i], g.graphOutputs[i]);
    };
    auto dispatch_graph = [&](const QnnSystemContext_GraphInfo_t& g) -> bool {
        switch (g.version) {
            case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: extract_graph(g.graphInfoV1); return true;
            case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: extract_graphV2(g.graphInfoV2); return true;
            case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: extract_graphV3(g.graphInfoV3); return true;
            default: LOGE("unknown graphInfo version=%d", (int)g.version); return false;
        }
    };

    const auto& info = *binInfo;
    if (info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        const auto& v1 = info.contextBinaryInfoV1;
        RETURN_IF(v1.numGraphs == 0, "binary has 0 graphs (v1)");
        if (!dispatch_graph(v1.graphs[0])) return false;
    } else if (info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        const auto& v2 = info.contextBinaryInfoV2;
        RETURN_IF(v2.numGraphs == 0, "binary has 0 graphs (v2)");
        if (!dispatch_graph(v2.graphs[0])) return false;
    } else if (info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        const auto& v3 = info.contextBinaryInfoV3;
        RETURN_IF(v3.numGraphs == 0, "binary has 0 graphs (v3)");
        if (!dispatch_graph(v3.graphs[0])) return false;
    } else {
        LOGE("unknown binaryInfo version %d", (int)info.version);
        return false;
    }

    LOGI("part %d: graph='%s' inputs=%zu outputs=%zu",
         part_idx, pi->graph_name.c_str(),
         pi->inputs.size(), pi->outputs.size());

    qsys_fns_->v.systemContextFree(sysCtx);

    // Allocate host backing storage for every input and output tensor.
    auto alloc = [&](TensorDesc& td, const std::string& tag) {
        const size_t bytes = td.total_elems * td.elem_bytes;
        // Disambiguate name to avoid input/output collisions if any.
        std::string key = tag + ":" + td.name;
        auto& buf = pi->storage[key];
        buf.assign(bytes, 0);
        td.host_buf = buf.data();
        SetClientBuf(td.tmpl, buf.data(), bytes);
    };
    for (auto& td : pi->inputs)  alloc(td, "in");
    for (auto& td : pi->outputs) alloc(td, "out");

    // Log tensor names for debugging.
    LOGI("part %d inputs:", part_idx);
    for (auto& td : pi->inputs) {
        LOGI("  IN  %-55s elems=%zu bytes=%zu", td.name.c_str(),
             td.total_elems, td.total_elems * td.elem_bytes);
    }
    LOGI("part %d outputs:", part_idx);
    for (auto& td : pi->outputs) {
        LOGI("  OUT %-55s elems=%zu bytes=%zu", td.name.c_str(),
             td.total_elems, td.total_elems * td.elem_bytes);
    }
    return true;
}

bool Gemma4Runner::RetrieveGraphHandlesForPart(int part_idx) {
    auto* pi = part_info_[part_idx].get();
    RETURN_IF(qnn_fns_->v.graphRetrieve(contexts_[part_idx],
                                        pi->graph_name.c_str(),
                                        &graphs_[part_idx]) != QNN_SUCCESS,
              "graphRetrieve failed");
    return true;
}

bool Gemma4Runner::CreateContextFromBinary(int part_idx) {
    const std::string& path = part_blob_paths_[part_idx];
    FILE* f = fopen(path.c_str(), "rb");
    RETURN_IF(!f, "cannot open context binary (lazy)");
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(sz);
    RETURN_IF(fread(blob.data(), 1, sz, f) != (size_t)sz, "read failed (lazy)");
    fclose(f);
    LOGI("part %d: loading context (%ld bytes) onto HTP", part_idx, sz);
    Qnn_ErrorHandle_t err = qnn_fns_->v.contextCreateFromBinary(
        backend_, device_, nullptr,
        blob.data(), blob.size(),
        &contexts_[part_idx], nullptr);
    if (err != QNN_SUCCESS) {
        LOGE("part %d: contextCreateFromBinary err=0x%llx", part_idx,
             (unsigned long long)err);
        return false;
    }
    return true;
}

void Gemma4Runner::FreeContext(int part_idx) {
    if (contexts_[part_idx] && qnn_fns_->v.contextFree) {
        qnn_fns_->v.contextFree(contexts_[part_idx], nullptr);
        contexts_[part_idx] = nullptr;
        graphs_[part_idx]   = nullptr;
    }
}

// ===========================================================================
// Chain execution
// ===========================================================================
const uint16_t* Gemma4Runner::RunChainOnce(const int32_t* seq_ids) {
    LOGI("=== RunChainOnce start ===");
    // Refresh the causal+padding mask for this prefill window.
    {
        auto it = rope_buffers_.find(
            "_text_model_layers_19_self_attn_Cast_9_output_0");
        if (it != rope_buffers_.end()) {
            FillCausalMaskFp16(reinterpret_cast<uint16_t*>(it->second.data()),
                               seq_ids, kPrefillSeq);
        }
    }
    // -- Run PLE on host: per_layer_inputs[1,32,35,256] fp16.
    // Determine the size: kPrefillSeq * kPleNumLayers * kPleDim * 2 bytes.
    // Stash directly into part0's per_layer_inputs input buffer (lookup by
    // name — the ONNX name for that tensor in part0 is "per_layer_inputs").
    auto find_input = [&](int p, const std::string& name) -> TensorDesc* {
        for (auto& td : part_info_[p]->inputs)
            if (td.name == name) return &td;
        return nullptr;
    };
    // The ONNX name for per_layer_inputs is "per_layer_inputs" on part0,
    // but on parts 1..3 the same tensor is passed through with a different
    // name: "/text_model/Mul_1_output_0" (the actual PLE-passthrough output
    // of part0). We look it up by name on each part.
    //
    // Convenience: for each part we copy the upstream output's host buffer
    // into the matching input's host buffer (by name), then execute.

    // -- Part 0: write input_ids and per_layer_inputs.
    // AI Hub may keep the original names ("input_ids", "per_layer_inputs")
    // OR rename them to generic forms. We match by shape first, then fall
    // back to name.
    auto find_input_by_shape = [&](int p, size_t elems, size_t elem_bytes) -> TensorDesc* {
        for (auto& td : part_info_[p]->inputs)
            if (td.total_elems == elems && td.elem_bytes == elem_bytes) return &td;
        return nullptr;
    };
    {
        TensorDesc* in_ids = find_input(0, "input_ids");
        if (!in_ids) in_ids = find_input_by_shape(0, kPrefillSeq, 4);  // int32
        if (!in_ids) in_ids = find_input_by_shape(0, kPrefillSeq, 8);  // int64
        TensorDesc* in_ple = find_input(0, "per_layer_inputs");
        if (!in_ple) {
            // (1, 32, 35, 256) fp16
            const size_t expected = (size_t)kPrefillSeq * 35 * 256;
            in_ple = find_input_by_shape(0, expected, 2);
        }
        if (!in_ids) { LOGE("part0: input ids tensor not found"); return nullptr; }
        if (!in_ple) { LOGE("part0: per_layer_inputs tensor not found"); return nullptr; }

        // Copy ids (truncate or cast int32 -> in_ids dtype which should be int32).
        // The export was 'int64' but compile flag --truncate_64bit_io makes it int32.
        if (in_ids->elem_bytes == 4) {
            std::memcpy(in_ids->host_buf, seq_ids, kPrefillSeq * 4);
        } else if (in_ids->elem_bytes == 8) {
            int64_t* dst = reinterpret_cast<int64_t*>(in_ids->host_buf);
            for (int i = 0; i < kPrefillSeq; ++i) dst[i] = (int64_t)seq_ids[i];
        } else {
            LOGE("input_ids has unsupported elem_bytes=%zu", in_ids->elem_bytes);
            return nullptr;
        }

        // PLE lookup -> per_layer_inputs.
        if (!ple_->Lookup(seq_ids, /*batch*/1, /*seq*/kPrefillSeq,
                          reinterpret_cast<uint16_t*>(in_ple->host_buf))) {
            LOGE("PLE lookup failed"); return nullptr;
        }
    }

    // Execute each part. We lazily load + free each part's context to fit
    // within the Hexagon PD memory budget (it can't hold all 5 GB of
    // weights at once).
    for (int p = 0; p < kNumParts; ++p) {
        auto* pi = part_info_[p].get();
        if (!CreateContextFromBinary(p))   { return nullptr; }
        if (!RetrieveGraphHandlesForPart(p)) { FreeContext(p); return nullptr; }

        // Build flat Qnn_Tensor_t arrays.
        std::vector<Qnn_Tensor_t> in_ts, out_ts;
        in_ts.reserve(pi->inputs.size());
        out_ts.reserve(pi->outputs.size());
        for (auto& td : pi->inputs)  in_ts.push_back(td.tmpl);
        for (auto& td : pi->outputs) out_ts.push_back(td.tmpl);

        Qnn_ErrorHandle_t err = qnn_fns_->v.graphExecute(
            graphs_[p],
            in_ts.data(),  (uint32_t)in_ts.size(),
            out_ts.data(), (uint32_t)out_ts.size(),
            nullptr, nullptr);
        if (err != QNN_SUCCESS) {
            LOGE("graphExecute(part %d) failed: 0x%llx", p,
                 (unsigned long long)err);
            FreeContext(p);
            return nullptr;
        }

        // Propagate outputs to next part's inputs. Strategy (in order):
        //   1. Explicit semantic mapping: parts 1/2/3 emit renamed outputs
        //      (output_0/output_1/...). We know what each *should* be from
        //      the source ONNX, so we hard-map them to the canonical names
        //      that downstream parts expect.
        //   2. Exact name match.
        //   3. Upstream search by name (for tensors that pass through
        //      multiple parts unchanged, e.g. per_layer_inputs).
        //   4. Host-computed RoPE injection.
        //   5. Shape-match fallback (warn — this is fuzzy).
        if (p + 1 < kNumParts) {
            auto* next = part_info_[p + 1].get();
            std::vector<bool> consumed_in(next->inputs.size(), false);

            auto copy_into = [&](size_t k, const void* src, size_t bytes) {
                auto& in_td = next->inputs[k];
                const size_t need_bytes = in_td.total_elems * in_td.elem_bytes;
                std::memcpy(in_td.host_buf, src,
                            bytes < need_bytes ? bytes : need_bytes);
                consumed_in[k] = true;
            };
            auto find_next_input = [&](const std::string& want) -> int {
                for (size_t k = 0; k < next->inputs.size(); ++k) {
                    if (consumed_in[k]) continue;
                    if (next->inputs[k].name == want) return (int)k;
                }
                return -1;
            };
            auto find_out_by_name = [&](const std::string& want) -> const TensorDesc* {
                for (auto& o : pi->outputs) if (o.name == want) return &o;
                return nullptr;
            };
            auto find_out_by_shape = [&](size_t elems, size_t bytes_per) -> const TensorDesc* {
                for (auto& o : pi->outputs) {
                    if (o.total_elems == elems && o.elem_bytes == bytes_per) return &o;
                }
                return nullptr;
            };

            // ---------------- (1) explicit semantic mapping ----------------
            // The 5-part split owns layers as: p0=[0..9], p1=[10..19],
            // p2=[20..27], p3=[28..34], p4=lm_head only. Each non-final part
            // emits the residual hidden state of its LAST layer (Mul_1) and,
            // for the KV-shared boundary layers (19 and 24), the K/V Casts.
            // Names are documented in exported_onnx_sha_split5/part{N}.onnx.
            struct Edge { const char* qnn_out; const char* canonical; };
            // From -> {QNN renamed output name on producer, canonical ONNX name}.
            // We use the canonical name on the consumer side.
            static const std::vector<std::vector<Edge>> kEdges = {
                // part 0 -> downstream  (output_0=hidden 49152, output_1=per_layer 286720)
                { {"output_0", "_text_model_layers_9_Mul_1_output_0"},
                  {"output_1", "_text_model_Mul_1_output_0"} },
                // part 1 -> downstream  (output_0=49152 hidden_19, output_1=16384 Cast,
                //                        output_2=16384 Cast_1; Cast_9 was dropped)
                { {"output_0", "_text_model_layers_19_Mul_1_output_0"},
                  {"output_1", "_text_model_layers_19_self_attn_Cast_output_0"},
                  {"output_2", "_text_model_layers_19_self_attn_Cast_1_output_0"} },
                // part 2 -> downstream  (output_0=49152 hidden_27, output_1=16384,
                //                        output_2=16384, output_3=1024)
                { {"output_0", "_text_model_layers_27_Mul_1_output_0"},
                  {"output_1", "_text_model_layers_24_self_attn_Cast_output_0"},
                  {"output_2", "_text_model_layers_24_self_attn_Cast_1_output_0"},
                  {"output_3", "_text_model_layers_24_self_attn_Cast_9_output_0"} },
                // part 3 -> part 4   (output_0 = norm output)
                { {"output_0", "_text_model_norm_Cast_output_0"} },
            };
            const auto& edges = kEdges[p];
            for (const auto& e : edges) {
                const TensorDesc* out = find_out_by_name(e.qnn_out);
                if (!out) continue;
                int k = find_next_input(e.canonical);
                if (k < 0) continue;  // downstream doesn't need it
                copy_into((size_t)k, out->host_buf,
                          out->total_elems * out->elem_bytes);
            }

            // ---------------- (2) exact name match -------------------------
            for (auto& out_td : pi->outputs) {
                if (out_td.name.empty()) continue;
                int k = find_next_input(out_td.name);
                if (k < 0) continue;
                auto& in_td = next->inputs[k];
                if (in_td.total_elems == out_td.total_elems &&
                    in_td.elem_bytes  == out_td.elem_bytes) {
                    copy_into((size_t)k, out_td.host_buf,
                              out_td.total_elems * out_td.elem_bytes);
                }
            }

            // ---------------- (3) upstream search by name ------------------
            for (size_t k = 0; k < next->inputs.size(); ++k) {
                if (consumed_in[k]) continue;
                auto& need = next->inputs[k];
                const TensorDesc* src = nullptr;
                for (int q = p; q >= 0 && !src; --q) {
                    for (auto& td : part_info_[q]->inputs) {
                        if (td.name == need.name &&
                            td.total_elems == need.total_elems &&
                            td.elem_bytes  == need.elem_bytes) { src = &td; break; }
                    }
                    if (src) break;
                    for (auto& td : part_info_[q]->outputs) {
                        if (td.name == need.name &&
                            td.total_elems == need.total_elems &&
                            td.elem_bytes  == need.elem_bytes) { src = &td; break; }
                    }
                }
                if (src) copy_into(k, src->host_buf,
                                   src->total_elems * src->elem_bytes);
            }

            // ---------------- (4) host-computed RoPE injection -------------
            for (size_t k = 0; k < next->inputs.size(); ++k) {
                if (consumed_in[k]) continue;
                auto& need = next->inputs[k];
                auto it = rope_buffers_.find(need.name);
                if (it == rope_buffers_.end()) continue;
                if (it->second.size() != need.total_elems * need.elem_bytes) {
                    LOGE("RoPE size mismatch for %s: have %zu need %zu",
                         need.name.c_str(), it->second.size(),
                         need.total_elems * need.elem_bytes);
                    continue;
                }
                copy_into(k, it->second.data(), it->second.size());
                LOGI("[part %d->%d] injected host RoPE '%s' (%zu bytes)",
                     p, p+1, need.name.c_str(), it->second.size());
            }

            // ---------------- (5) shape fallback ---------------------------
            // Only used for tensors we still can't identify. Warns to log.
            for (auto& out_td : pi->outputs) {
                // skip outputs already used by semantic mapping; the
                // simplest check is by name.
                bool was_mapped = false;
                for (const auto& e : edges) {
                    if (out_td.name == e.qnn_out) { was_mapped = true; break; }
                }
                if (was_mapped) continue;
                const TensorDesc* match_in = nullptr;
                int matched_k = -1;
                for (size_t k = 0; k < next->inputs.size(); ++k) {
                    if (consumed_in[k]) continue;
                    auto& in_td = next->inputs[k];
                    if (in_td.total_elems == out_td.total_elems &&
                        in_td.elem_bytes  == out_td.elem_bytes) {
                        match_in = &in_td; matched_k = (int)k; break;
                    }
                }
                if (match_in && matched_k >= 0) {
                    copy_into((size_t)matched_k, out_td.host_buf,
                              out_td.total_elems * out_td.elem_bytes);
                    LOGI("[part %d->%d] fuzzy shape-map '%s' -> '%s' (%zu elems)",
                         p, p+1, out_td.name.c_str(),
                         match_in->name.c_str(), out_td.total_elems);
                }
            }

            // Final report of un-matched inputs (zero-filled).
            for (size_t k = 0; k < next->inputs.size(); ++k) {
                if (consumed_in[k]) {
                    LOGI("[part %d->%d] bound '%s' (%zu elems)",
                         p, p+1, next->inputs[k].name.c_str(),
                         next->inputs[k].total_elems);
                } else {
                    LOGE("[part %d->%d] UN-MATCHED '%s' (elems=%zu) -> zeros",
                         p, p+1, next->inputs[k].name.c_str(),
                         next->inputs[k].total_elems);
                }
            }
        }
        // Free this part's HTP context (PD memory budget).
        FreeContext(p);
    }

    // Last part: locate the logits tensor. AI Hub may rename "logits" to
    // "output_0"; we pick the largest-elem output (vocab=262144 >> 1536).
    auto* last_pi = part_info_[kNumParts - 1].get();
    TensorDesc* logits = nullptr;
    size_t best_elems = 0;
    for (auto& td : last_pi->outputs) {
        if (td.total_elems > best_elems) { best_elems = td.total_elems; logits = &td; }
    }
    if (!logits) {
        LOGE("part %d: no outputs", kNumParts - 1);
        return nullptr;
    }
    return reinterpret_cast<const uint16_t*>(logits->host_buf);
}

// ===========================================================================
// FP16 -> FP32 + sampling
// ===========================================================================
static inline float HalfToFloat(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h & 0x7C00u) >> 10;
    const uint32_t frac = (h & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
        f = sign | (frac ? ((frac << 13) | 0x38000000u) : 0u);
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (frac << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (frac << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

int32_t Gemma4Runner::SampleAtPosition(const uint16_t* logits_window, int pos) {
    const uint16_t* row = logits_window + (size_t)pos * kVocabSize;
    int best = 0;
    float bestv = HalfToFloat(row[0]);
    for (int i = 1; i < kVocabSize; ++i) {
        const float v = HalfToFloat(row[i]);
        if (v > bestv) { bestv = v; best = i; }
    }
    return best;
}

// ===========================================================================
// Generation
// ===========================================================================
std::vector<int32_t> Gemma4Runner::GenerateFromIds(
        const std::vector<int32_t>& context_ids,
        int max_new_tokens,
        const TokenSink& on_token) {
    std::vector<int32_t> generated;
    LOGI("GenerateFromIds: context_ids=%zu max_new_tokens=%d",
         context_ids.size(), max_new_tokens);
    if (context_ids.empty()) return generated;

    // Rolling 32-token window. Right-padded with PAD (0) until full.
    std::vector<int32_t> window(kPrefillSeq, 0);
    int num_real = (int)std::min<size_t>(context_ids.size(), kPrefillSeq);
    std::copy(context_ids.begin(), context_ids.begin() + num_real,
              window.begin());

    for (int step = 0; step < max_new_tokens; ++step) {
        const uint16_t* logits = RunChainOnce(window.data());
        if (!logits) {
            LOGE("RunChainOnce returned null at step %d", step);
            break;
        }
        const int pos = num_real - 1;  // last *real* token
        const int32_t next = SampleAtPosition(logits, pos);
        generated.push_back(next);
        if (on_token) on_token(next, step);

        // Append to window. If full, shift left by 1.
        if (num_real < kPrefillSeq) {
            window[num_real] = next;
            num_real++;
        } else {
            // Shift left; place new token at slot 31.
            std::memmove(&window[0], &window[1], (kPrefillSeq - 1) * sizeof(int32_t));
            window[kPrefillSeq - 1] = next;
        }
    }
    return generated;
}

std::string Gemma4Runner::Generate(const std::string& prompt) {
    if (!tokenizer_) return {};
    auto ids = tokenizer_->Encode(prompt, /*add_bos*/true);
    auto out = GenerateFromIds(ids, opts_.max_new_tokens, nullptr);
    return tokenizer_->Decode(out);
}

}  // namespace gemma4
