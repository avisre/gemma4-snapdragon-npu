// Per-Layer Embeddings (PLE) preprocessor for Gemma 4 E2B on Hexagon v69.
//
// The 4.7 GB PLE table is externalized from the NPU graph: the NPU takes
// per_layer_inputs of shape (batch, seq_len, num_layers, ple_dim) as an
// input, and CPU computes it via a gather + reshape over the PLE table
// before each NPU invocation.
//
// File format ("PLE1") matches ple_preprocess.py exactly. See that file
// for the full layout description and the formula derivation.

#ifndef GEMMA4_RUNTIME_PLE_PREPROCESS_H_
#define GEMMA4_RUNTIME_PLE_PREPROCESS_H_

#include <cstdint>
#include <cstddef>
#include <string>

namespace gemma4 {

// Gemma 4 E2B constants (from config.json).
constexpr uint32_t kPleVocabSize = 262144;
constexpr uint32_t kPleNumLayers = 35;
constexpr uint32_t kPleDim       = 256;
constexpr uint32_t kPlePackedDim = kPleNumLayers * kPleDim;   // 8960

// NPU graph shape (matches what we compiled for v69).
constexpr uint32_t kPleBatch  = 1;
constexpr uint32_t kPleSeqLen = 128;

// File-format dtype IDs (must match ple_preprocess.py).
enum PleDtype : uint32_t {
  kPleDtypeFp16 = 1,
  kPleDtypeBf16 = 2,
  kPleDtypeFp32 = 3,
};

#pragma pack(push, 1)
struct PleHeader {
  char     magic[4];          // "PLE1"
  uint32_t vocab_size;        // 262144
  uint32_t num_layers;        // 35
  uint32_t ple_dim;           // 256
  uint32_t dtype;             // PleDtype
  uint32_t embed_scale_baked; // 1 if sqrt(ple_dim) already multiplied in
};
#pragma pack(pop)
static_assert(sizeof(PleHeader) == 24, "PleHeader must be 24 bytes");

// Owns an mmap of the packed PLE binary and performs gather+reshape lookups.
//
// Thread-safety: Load() is not thread-safe, Lookup() is read-only and safe to
// call concurrently from multiple threads once Load() has returned.
class PLEPreprocessor {
 public:
  PLEPreprocessor() = default;
  ~PLEPreprocessor();

  // Disable copy; allow move.
  PLEPreprocessor(const PLEPreprocessor&) = delete;
  PLEPreprocessor& operator=(const PLEPreprocessor&) = delete;
  PLEPreprocessor(PLEPreprocessor&&) noexcept;
  PLEPreprocessor& operator=(PLEPreprocessor&&) noexcept;

  // mmap the packed PLE binary. Returns false on error (file missing,
  // bad magic, dtype not supported, etc.). On Android, `path` is typically
  // under the app's filesDir (e.g. /data/data/<pkg>/files/ple_weights.bin).
  bool Load(const std::string& path);

  // Gather + reshape.
  //   input_ids : pointer to [batch * seq_len] int32 token ids
  //   out       : pointer to [batch * seq_len * num_layers * ple_dim] fp16 buffer
  //               (caller-allocated; layout is row-major NSHd)
  // Returns false on out-of-range id or shape mismatch with the loaded table.
  // Currently only fp16 weights are supported on the device path.
  bool Lookup(const int32_t* input_ids,
              uint32_t batch,
              uint32_t seq_len,
              uint16_t* out) const;

  // Convenience wrapper for the compiled NPU shape (batch=1, seq_len=128).
  bool LookupFixed(const int32_t* input_ids, uint16_t* out) const {
    return Lookup(input_ids, kPleBatch, kPleSeqLen, out);
  }

  // Accessors (after a successful Load()).
  uint32_t vocab_size() const { return header_.vocab_size; }
  uint32_t num_layers() const { return header_.num_layers; }
  uint32_t ple_dim()    const { return header_.ple_dim; }
  PleDtype dtype()      const { return static_cast<PleDtype>(header_.dtype); }
  bool     embed_scale_baked() const { return header_.embed_scale_baked != 0; }
  size_t   bytes_per_row() const;

 private:
  void Reset();

  PleHeader   header_{};
  int         fd_       = -1;
  void*       mapping_  = nullptr;   // start of the whole file
  size_t      map_size_ = 0;
  const void* weights_  = nullptr;   // mapping_ + sizeof(PleHeader)
};

}  // namespace gemma4

#endif  // GEMMA4_RUNTIME_PLE_PREPROCESS_H_
