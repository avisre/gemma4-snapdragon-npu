// gemma4_main.cpp — CLI entry point for the multi-part Gemma 4 E2B runner.
//
// Modes:
//   --probe                : dlopen + provider check (legacy smoke test)
//   --bin <path>           : stat-check a single context binary
//   --model_paths a,b,...  : list of part .bin paths (comma-separated)
//   --ple_path <path>      : path to packed PLE binary (PLE1 header)
//   --input_ids_path <p>   : binary file of int32 token ids (length L<=32)
//   --max_tokens N         : number of tokens to generate (default 16)
//   --qnn_sdk_root <p>     : path to QNN SDK (fallback for lib lookup)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnLog.h"
#include "QnnCommon.h"
#include "System/QnnSystemInterface.h"

#include "gemma4_runner.h"
#include "ple_preprocess.h"

namespace {

void PrintHelp(const char* argv0) {
    std::printf(
        "gemma4_runner — Gemma-4 E2B on Hexagon v69 (QNN)\n"
        "\nUsage:\n"
        "  %s --probe\n"
        "  %s --model_paths p0.bin,p1.bin,p2.bin,p3.bin,p4.bin \\\n"
        "     --ple_path packed.ple --input_ids_path ids.bin --max_tokens 16\n",
        argv0, argv0);
}

std::vector<std::string> SplitCsv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            if (i > start) out.emplace_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

int Probe() {
    void* htp = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (!htp) { std::fprintf(stderr, "dlopen htp: %s\n", dlerror()); return 1; }
    void* sys = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (!sys) { std::fprintf(stderr, "dlopen sys: %s\n", dlerror()); return 2; }
    std::printf("[probe] OK htp=%p sys=%p\n", htp, sys);
    dlclose(sys); dlclose(htp);
    return 0;
}

bool ReadInt32Bin(const std::string& path, std::vector<int32_t>* out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "[main] open %s failed\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz % 4 != 0) { std::fclose(f); return false; }
    out->resize(sz / 4);
    if (std::fread(out->data(), 1, sz, f) != (size_t)sz) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { PrintHelp(argv[0]); return 0; }

    std::string model_paths_csv, ple_path, input_ids_path, qnn_sdk_root;
    int  max_tokens = 16;
    bool did_probe = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { PrintHelp(argv[0]); return 0; }
        if (a == "--probe")               { if (Probe() != 0) return 1; did_probe = true; continue; }
        if (a == "--model_paths" && i+1 < argc) { model_paths_csv = argv[++i]; continue; }
        if (a == "--ple_path"    && i+1 < argc) { ple_path        = argv[++i]; continue; }
        if (a == "--input_ids_path" && i+1 < argc) { input_ids_path = argv[++i]; continue; }
        if (a == "--max_tokens"  && i+1 < argc) { max_tokens      = std::atoi(argv[++i]); continue; }
        if (a == "--qnn_sdk_root" && i+1 < argc) { qnn_sdk_root   = argv[++i]; continue; }
        std::fprintf(stderr, "[main][W] unknown arg: %s\n", a.c_str());
    }

    if (did_probe && model_paths_csv.empty()) return 0;
    if (model_paths_csv.empty() || ple_path.empty() || input_ids_path.empty()) {
        PrintHelp(argv[0]);
        return 1;
    }

    auto paths = SplitCsv(model_paths_csv);
    if ((int)paths.size() != gemma4::kNumParts) {
        std::fprintf(stderr, "[main] need exactly %d model paths (got %zu)\n",
                     gemma4::kNumParts, paths.size());
        return 1;
    }

    // Load PLE.
    gemma4::PLEPreprocessor ple;
    if (!ple.Load(ple_path)) {
        std::fprintf(stderr, "[main] PLE load failed: %s\n", ple_path.c_str());
        return 1;
    }
    std::printf("[main] PLE loaded: %s\n", ple_path.c_str());

    // Load input ids.
    std::vector<int32_t> ids;
    if (!ReadInt32Bin(input_ids_path, &ids)) {
        std::fprintf(stderr, "[main] input_ids read failed\n"); return 1;
    }
    std::printf("[main] input_ids (%zu):", ids.size());
    for (size_t i = 0; i < ids.size() && i < 16; ++i) std::printf(" %d", ids[i]);
    if (ids.size() > 16) std::printf(" ...");
    std::printf("\n");

    // Initialize runner.
    gemma4::Gemma4Runner runner;
    gemma4::Gemma4Runner::Options opts;
    opts.qnn_sdk_root         = qnn_sdk_root;
    opts.context_binary_paths = paths;
    opts.max_new_tokens       = max_tokens;
    if (!runner.Initialize(opts, /*tokenizer*/nullptr, &ple)) {
        std::fprintf(stderr, "[main] runner init failed\n"); return 2;
    }
    std::printf("[main] runner initialized; generating %d tokens...\n", max_tokens);

    // Generate and stream token ids to stdout.
    std::fprintf(stderr, "[main] passing %zu ids to GenerateFromIds, max=%d\n",
                 ids.size(), max_tokens);
    auto out = runner.GenerateFromIds(ids, max_tokens,
        [](int32_t tok, int step) {
            std::printf("[tok %2d] id=%d\n", step, tok);
            std::fflush(stdout);
        });

    std::printf("[main] DONE — generated %zu tokens: ", out.size());
    for (auto t : out) std::printf("%d ", t);
    std::printf("\n");
    return 0;
}
