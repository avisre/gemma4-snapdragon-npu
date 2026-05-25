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
    return true;
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

        // Propagate outputs to next part's inputs by NAME first (AI Hub
        // sometimes preserves the original ONNX names for cross-part
        // edges), then by shape as a fallback.
        if (p + 1 < kNumParts) {
            auto* next = part_info_[p + 1].get();
            std::vector<bool> consumed_in(next->inputs.size(), false);
            // Pass 1: name-exact match.
            for (auto& out_td : pi->outputs) {
                if (out_td.name.empty()) continue;
                for (size_t k = 0; k < next->inputs.size(); ++k) {
                    if (consumed_in[k]) continue;
                    auto& in_td = next->inputs[k];
                    if (in_td.name == out_td.name &&
                        in_td.total_elems * in_td.elem_bytes ==
                        out_td.total_elems * out_td.elem_bytes) {
                        std::memcpy(in_td.host_buf, out_td.host_buf,
                                    in_td.total_elems * in_td.elem_bytes);
                        consumed_in[k] = true;
                        break;
                    }
                }
            }
            // Pass 2: shape match for the rest.
            for (auto& out_td : pi->outputs) {
                int matched = -1;
                for (size_t k = 0; k < next->inputs.size(); ++k) {
                    if (consumed_in[k]) continue;
                    auto& in_td = next->inputs[k];
                    if (in_td.total_elems == out_td.total_elems &&
                        in_td.elem_bytes  == out_td.elem_bytes) {
                        matched = (int)k; break;
                    }
                }
                if (matched >= 0) {
                    auto& in_td = next->inputs[matched];
                    const size_t bytes = in_td.total_elems * in_td.elem_bytes;
                    std::memcpy(in_td.host_buf, out_td.host_buf, bytes);
                    consumed_in[matched] = true;
                }
            }
            // For inputs that still didn't match (e.g. attention scratch
            // tensors that part0 dropped but part_N+1 still needs), look
            // them up by NAME in *upstream* parts' outputs/inputs and
            // forward those.
            for (size_t k = 0; k < next->inputs.size(); ++k) {
                if (consumed_in[k]) continue;
                auto& need = next->inputs[k];
                // Search upstream parts q < p+1 for a same-name tensor.
                const TensorDesc* src = nullptr;
                for (int q = p; q >= 0 && !src; --q) {
                    for (auto& td : part_info_[q]->inputs)
                        if (td.name == need.name &&
                            td.total_elems == need.total_elems &&
                            td.elem_bytes  == need.elem_bytes) { src = &td; break; }
                    if (src) break;
                    for (auto& td : part_info_[q]->outputs)
                        if (td.name == need.name &&
                            td.total_elems == need.total_elems &&
                            td.elem_bytes  == need.elem_bytes) { src = &td; break; }
                }
                if (src) {
                    std::memcpy(need.host_buf, src->host_buf,
                                need.total_elems * need.elem_bytes);
                    consumed_in[k] = true;
                } else {
                    LOGI("[part %d->%d] input '%s' (elems=%zu) un-matched, zero",
                         p, p+1, need.name.c_str(), need.total_elems);
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
