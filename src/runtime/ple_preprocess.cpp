// Per-Layer Embeddings (PLE) preprocessor: Android runtime implementation.
//
// Build (NDK r26+, arm64-v8a):
//
//   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang++
//       -std=c++17 -O3 -fPIC
//       -c runtime/ple_preprocess.cpp -o ple_preprocess.o
//
// or via CMake (suggested CMakeLists entry):
//
//   add_library(ple_preprocess STATIC runtime/ple_preprocess.cpp)
//   target_include_directories(ple_preprocess PUBLIC runtime)
//   target_compile_features(ple_preprocess PUBLIC cxx_std_17)
//
// On Hexagon HVX you can later swap the inner row-copy for an HVX vector copy;
// the current implementation is a portable scalar memcpy that the compiler
// already auto-vectorizes well on ARMv8.

#include "ple_preprocess.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>

namespace gemma4 {

namespace {

constexpr char kMagic[4] = {'P', 'L', 'E', '1'};

size_t DtypeSize(PleDtype d) {
  switch (d) {
    case kPleDtypeFp16: return 2;
    case kPleDtypeBf16: return 2;
    case kPleDtypeFp32: return 4;
  }
  return 0;
}

}  // namespace

PLEPreprocessor::~PLEPreprocessor() { Reset(); }

PLEPreprocessor::PLEPreprocessor(PLEPreprocessor&& other) noexcept { *this = std::move(other); }

PLEPreprocessor& PLEPreprocessor::operator=(PLEPreprocessor&& other) noexcept {
  if (this != &other) {
    Reset();
    header_   = other.header_;
    fd_       = other.fd_;
    mapping_  = other.mapping_;
    map_size_ = other.map_size_;
    weights_  = other.weights_;
    other.fd_       = -1;
    other.mapping_  = nullptr;
    other.map_size_ = 0;
    other.weights_  = nullptr;
    std::memset(&other.header_, 0, sizeof(other.header_));
  }
  return *this;
}

void PLEPreprocessor::Reset() {
  if (mapping_ != nullptr && mapping_ != MAP_FAILED) {
    ::munmap(mapping_, map_size_);
  }
  if (fd_ >= 0) {
    ::close(fd_);
  }
  fd_       = -1;
  mapping_  = nullptr;
  map_size_ = 0;
  weights_  = nullptr;
  std::memset(&header_, 0, sizeof(header_));
}

size_t PLEPreprocessor::bytes_per_row() const {
  return static_cast<size_t>(header_.num_layers) * header_.ple_dim *
         DtypeSize(static_cast<PleDtype>(header_.dtype));
}

bool PLEPreprocessor::Load(const std::string& path) {
  Reset();

  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) {
    std::fprintf(stderr, "[ple] open(%s) failed: %s\n", path.c_str(), std::strerror(errno));
    return false;
  }

  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    std::fprintf(stderr, "[ple] fstat failed: %s\n", std::strerror(errno));
    Reset();
    return false;
  }
  map_size_ = static_cast<size_t>(st.st_size);
  if (map_size_ < sizeof(PleHeader)) {
    std::fprintf(stderr, "[ple] file too small (%zu bytes)\n", map_size_);
    Reset();
    return false;
  }

  mapping_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (mapping_ == MAP_FAILED) {
    std::fprintf(stderr, "[ple] mmap failed: %s\n", std::strerror(errno));
    mapping_ = nullptr;
    Reset();
    return false;
  }

  // Hint sequential / random access patterns (gather is random across rows).
  ::madvise(mapping_, map_size_, MADV_RANDOM);

  std::memcpy(&header_, mapping_, sizeof(header_));
  if (std::memcmp(header_.magic, kMagic, 4) != 0) {
    std::fprintf(stderr, "[ple] bad magic\n");
    Reset();
    return false;
  }
  if (header_.dtype != kPleDtypeFp16) {
    // Only fp16 is supported on-device. BF16/FP32 would need a runtime cast.
    std::fprintf(stderr, "[ple] unsupported on-device dtype %u (need fp16)\n", header_.dtype);
    Reset();
    return false;
  }
  if (header_.vocab_size != kPleVocabSize || header_.num_layers != kPleNumLayers ||
      header_.ple_dim != kPleDim) {
    std::fprintf(stderr,
                 "[ple] shape mismatch: got (%u,%u,%u), expected (%u,%u,%u)\n",
                 header_.vocab_size, header_.num_layers, header_.ple_dim,
                 kPleVocabSize, kPleNumLayers, kPleDim);
    Reset();
    return false;
  }

  const size_t expected =
      sizeof(PleHeader) + static_cast<size_t>(header_.vocab_size) * bytes_per_row();
  if (map_size_ < expected) {
    std::fprintf(stderr, "[ple] file truncated: have %zu, need %zu\n", map_size_, expected);
    Reset();
    return false;
  }

  weights_ = static_cast<const uint8_t*>(mapping_) + sizeof(PleHeader);
  return true;
}

bool PLEPreprocessor::Lookup(const int32_t* input_ids,
                             uint32_t batch,
                             uint32_t seq_len,
                             uint16_t* out) const {
  if (weights_ == nullptr || input_ids == nullptr || out == nullptr) {
    return false;
  }
  if (header_.dtype != kPleDtypeFp16) {
    return false;  // device path is fp16-only
  }

  // Row stride in elements (num_layers * ple_dim = 8960).
  const size_t row_elems = static_cast<size_t>(header_.num_layers) * header_.ple_dim;
  const size_t row_bytes = row_elems * sizeof(uint16_t);
  const uint16_t* table = static_cast<const uint16_t*>(weights_);
  const uint32_t  vocab = header_.vocab_size;

  // Output layout (NSHd): out[b, s, l, d] is contiguous over (l, d) for a
  // given (b, s) — exactly the same layout as a row of the PLE table —
  // so the per-token operation degenerates to a single memcpy.
  const size_t total_tokens = static_cast<size_t>(batch) * seq_len;
  for (size_t i = 0; i < total_tokens; ++i) {
    const int32_t id = input_ids[i];
    if (id < 0 || static_cast<uint32_t>(id) >= vocab) {
      std::fprintf(stderr, "[ple] token id %d out of range [0,%u)\n", id, vocab);
      return false;
    }
    const uint16_t* src = table + static_cast<size_t>(id) * row_elems;
    uint16_t*       dst = out   + i                       * row_elems;
    std::memcpy(dst, src, row_bytes);
  }
  return true;
}

}  // namespace gemma4
