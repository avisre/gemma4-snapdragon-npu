// runtime/tokenizer.cpp
// SentencePiece-backed implementation of gemma4::Tokenizer.

#include "tokenizer.h"

#include <android/log.h>
#include <sentencepiece_processor.h>

#include <cstring>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "gemma4-tok", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "gemma4-tok", __VA_ARGS__)

namespace gemma4 {

Tokenizer::Tokenizer()
    : sp_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {}

Tokenizer::~Tokenizer() = default;

bool Tokenizer::Load(const std::string& model_path) {
  const auto status = sp_->Load(model_path);
  if (!status.ok()) {
    LOGE("SentencePiece load failed: %s", status.ToString().c_str());
    return false;
  }
  if (sp_->GetPieceSize() != kVocabSize) {
    LOGE("Vocab mismatch: got %d expected %d",
         sp_->GetPieceSize(), kVocabSize);
    // Don't hard-fail: some checkpoints ship slightly trimmed vocabs.
  }
  LOGI("Loaded SentencePiece model: vocab=%d", sp_->GetPieceSize());
  return true;
}

std::vector<int32_t> Tokenizer::Encode(const std::string& text,
                                       bool add_bos,
                                       bool add_eos) const {
  std::vector<int> ids;
  sp_->Encode(text, &ids);
  std::vector<int32_t> out;
  out.reserve(ids.size() + 2);
  if (add_bos) out.push_back(kBosId);
  for (int v : ids) out.push_back(static_cast<int32_t>(v));
  if (add_eos) out.push_back(kEosId);
  return out;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& tokens) const {
  std::vector<int> ids(tokens.begin(), tokens.end());
  std::string out;
  sp_->Decode(ids, &out);
  return out;
}

std::string Tokenizer::DecodePiece(int32_t token) const {
  return sp_->IdToPiece(static_cast<int>(token));
}

}  // namespace gemma4
