// runtime/tokenizer.h
// Gemma-4 E2B SentencePiece tokenizer wrapper for Android.
// Vocab: 262,144. Special tokens: PAD=0, EOS=1, BOS=2, UNK=3.
//
// The on-device tokenizer file must be the raw SentencePiece model
// (tokenizer.model). The HF tokenizer.json is NOT consumed by this class.
// If only tokenizer.json is available at training time, convert with:
//   python -c "from transformers import AutoTokenizer;
//              AutoTokenizer.from_pretrained('checkpoints/gemma-4-e2b-it')
//                           .save_vocabulary('runtime/')"
// which emits runtime/tokenizer.model.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sentencepiece { class SentencePieceProcessor; }

namespace gemma4 {

class Tokenizer {
 public:
  // Special-token ids (canonical for Gemma-4 E2B).
  static constexpr int32_t kPadId = 0;
  static constexpr int32_t kEosId = 1;
  static constexpr int32_t kBosId = 2;
  static constexpr int32_t kUnkId = 3;
  static constexpr int32_t kVocabSize = 262144;

  Tokenizer();
  ~Tokenizer();

  // Load the SentencePiece model from disk. Returns true on success.
  // `model_path` typically points to /data/local/tmp/runtime/tokenizer.model
  // when pushed via adb, or to an asset path under the Android app.
  bool Load(const std::string& model_path);

  // Encode UTF-8 `text` into token ids. If `add_bos`, prepends BOS.
  // If `add_eos`, appends EOS.
  std::vector<int32_t> Encode(const std::string& text,
                              bool add_bos = true,
                              bool add_eos = false) const;

  // Decode token ids back to UTF-8 text.
  std::string Decode(const std::vector<int32_t>& tokens) const;

  // Convenience: single-token decode (used during streaming).
  std::string DecodePiece(int32_t token) const;

  int32_t bos_id() const { return kBosId; }
  int32_t eos_id() const { return kEosId; }
  int32_t pad_id() const { return kPadId; }
  int32_t unk_id() const { return kUnkId; }
  int32_t vocab_size() const { return kVocabSize; }

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
};

}  // namespace gemma4
