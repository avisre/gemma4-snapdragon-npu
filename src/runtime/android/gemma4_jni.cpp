// gemma4_jni.cpp
//
// JNI bridge for Gemma 4 E2B on Android (Hexagon v69 via QNN HTP). Sits
// between the Kotlin `ai.localyze.runtime.Gemma4Runtime` facade and the
// C++ `gemma4::Gemma4Runner` + `gemma4::PLEPreprocessor` runtime.
//
// Build: see android/CMakeLists.txt (to be added alongside this file).
// ARM64 only, NDK r26+, C++17.
//
// Pattern mirrors the existing Qwen3 setup at runtime/jni/, but exposes the
// simplified Gemma4Runtime API requested by the Localyze app
// (constructor + generate(streaming) + close).

#include "gemma4_jni.h"

#include <android/log.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Native runtime headers (siblings of this file, two levels up).
#include "../gemma4_runner.h"
#include "../ple_preprocess.h"
#include "../tokenizer.h"

#define LOG_TAG "Gemma4JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// ---------------------------------------------------------------------------
// RunnerContext — owns every native resource that backs one Kotlin object.
// All field access is serialized via `mu`; concurrent JNI calls on the same
// handle are queued by the Kotlin side (and additionally guarded here).
// ---------------------------------------------------------------------------
struct RunnerContext {
    std::mutex                                 mu;
    std::unique_ptr<gemma4::Gemma4Runner>      runner;
    std::unique_ptr<gemma4::PLEPreprocessor>   ple;
    std::unique_ptr<gemma4::ITokenizer>        tokenizer;
    JavaVM*                                    jvm   = nullptr;
    std::atomic<bool>                          ready{false};
    std::atomic<bool>                          cancel{false};

    // Cached paths (debug/logging only).
    std::string bin_path;
    std::string ple_path;
    std::string tok_path;
};

inline RunnerContext* CtxFromHandle(jlong h) {
    return reinterpret_cast<RunnerContext*>(static_cast<uintptr_t>(h));
}

inline jlong HandleFromCtx(RunnerContext* c) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(c));
}

// jstring -> std::string (UTF-8). Null-safe.
std::string JStrToStd(JNIEnv* env, jstring js) {
    if (!js) return {};
    const char* c = env->GetStringUTFChars(js, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(js, c);
    return out;
}

// Ensure the calling thread is attached to the JVM and invoke `fn(env)`.
// Detaches only if we attached it ourselves. Use for worker-thread callbacks.
template <typename F>
void WithEnv(JavaVM* jvm, F&& fn) {
    JNIEnv* env  = nullptr;
    bool attached = false;
    if (!jvm) return;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (jvm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
            LOGE("AttachCurrentThread failed");
            return;
        }
        attached = true;
    }
    fn(env);
    if (attached) jvm->DetachCurrentThread();
}

}  // namespace

// ===========================================================================
// JNI entry points
// ===========================================================================
extern "C" {

// ---------------------------------------------------------------------------
// nativeCreate
// ---------------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeCreate(
        JNIEnv* env, jobject /*thiz*/) {
    auto* ctx = new (std::nothrow) RunnerContext();
    if (!ctx) {
        LOGE("nativeCreate: OOM");
        return 0L;
    }
    env->GetJavaVM(&ctx->jvm);
    LOGI("nativeCreate handle=%p", static_cast<void*>(ctx));
    return HandleFromCtx(ctx);
}

// ---------------------------------------------------------------------------
// nativeInit
// ---------------------------------------------------------------------------
JNIEXPORT jboolean JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeInit(
        JNIEnv* env, jobject /*thiz*/, jlong handle,
        jstring binPath, jstring pleBinPath, jstring tokenizerPath) {

    auto* ctx = CtxFromHandle(handle);
    if (!ctx) {
        LOGE("nativeInit: null handle");
        return JNI_FALSE;
    }

    std::lock_guard<std::mutex> lk(ctx->mu);
    if (ctx->ready.load()) {
        LOGW("nativeInit: already initialized; ignoring");
        return JNI_TRUE;
    }

    ctx->bin_path = JStrToStd(env, binPath);
    ctx->ple_path = JStrToStd(env, pleBinPath);
    ctx->tok_path = JStrToStd(env, tokenizerPath);

    LOGI("nativeInit bin=%s ple=%s tok=%s",
         ctx->bin_path.c_str(), ctx->ple_path.c_str(), ctx->tok_path.c_str());

    try {
        // 1) PLE preprocessor — mmap the 4.7 GB external embedding table.
        ctx->ple = std::make_unique<gemma4::PLEPreprocessor>();
        if (!ctx->ple->Load(ctx->ple_path)) {
            LOGE("PLEPreprocessor::Load failed: %s", ctx->ple_path.c_str());
            ctx->ple.reset();
            return JNI_FALSE;
        }

        // 2) Tokenizer — Gemma SentencePiece, 262 144 vocab. Concrete impl
        //    lives in runtime/tokenizer.cpp; factory returns the abstract
        //    ITokenizer* so the JNI layer doesn't depend on the backend.
        ctx->tokenizer.reset(gemma4::CreateSentencePieceTokenizer(ctx->tok_path));
        if (!ctx->tokenizer) {
            LOGE("Tokenizer load failed: %s", ctx->tok_path.c_str());
            ctx->ple.reset();
            return JNI_FALSE;
        }

        // 3) QNN runner. dlopen of libQnnHtp.so happens inside Initialize().
        gemma4::Gemma4Runner::Options opts;
        opts.context_binary_path = ctx->bin_path;
        opts.backend_lib         = "libQnnHtp.so";   // shipped in APK jniLibs
        opts.system_lib          = "libQnnSystem.so";
        opts.prefill_graph_name  = "prefill_128";
        opts.decode_graph_name   = "decode_1";

        ctx->runner = std::make_unique<gemma4::Gemma4Runner>();
        if (!ctx->runner->Initialize(opts, ctx->tokenizer.get(), ctx->ple.get())) {
            LOGE("Gemma4Runner::Initialize failed");
            ctx->runner.reset();
            ctx->tokenizer.reset();
            ctx->ple.reset();
            return JNI_FALSE;
        }

        ctx->ready.store(true);
        LOGI("nativeInit ok");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("nativeInit exception: %s", e.what());
        ctx->runner.reset();
        ctx->tokenizer.reset();
        ctx->ple.reset();
        ctx->ready.store(false);
        return JNI_FALSE;
    }
}

// ---------------------------------------------------------------------------
// nativeGenerate — prefill + streaming decode.
// ---------------------------------------------------------------------------
JNIEXPORT jstring JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeGenerate(
        JNIEnv* env, jobject /*thiz*/, jlong handle,
        jstring prompt, jint maxTokens, jfloat temperature,
        jobject onTokenLambda) {

    auto* ctx = CtxFromHandle(handle);
    if (!ctx || !ctx->ready.load()) {
        LOGE("nativeGenerate: not ready");
        return env->NewStringUTF("");
    }

    const std::string prompt_str = JStrToStd(env, prompt);
    if (prompt_str.empty()) {
        return env->NewStringUTF("");
    }

    // Cache Function1.invoke. Hold a global ref so we can call back from the
    // worker thread context (current thread for this synchronous impl).
    jclass    lambdaCls = onTokenLambda ? env->GetObjectClass(onTokenLambda) : nullptr;
    jmethodID invokeMid = lambdaCls
            ? env->GetMethodID(lambdaCls, "invoke",
                               "(Ljava/lang/Object;)Ljava/lang/Object;")
            : nullptr;
    jobject lambdaGlobal = onTokenLambda ? env->NewGlobalRef(onTokenLambda) : nullptr;

    ctx->cancel.store(false);
    std::string accumulated;

    // Streaming callback: invoked by the runner each time a new piece is
    // detokenized. May arrive from the same thread (current decode model)
    // or a worker thread — WithEnv handles both.
    auto emit = [&](const std::string& piece) {
        accumulated.append(piece);
        if (lambdaGlobal && invokeMid) {
            WithEnv(ctx->jvm, [&](JNIEnv* tenv) {
                jstring js = tenv->NewStringUTF(piece.c_str());
                tenv->CallObjectMethod(lambdaGlobal, invokeMid, js);
                tenv->DeleteLocalRef(js);
                if (tenv->ExceptionCheck()) {
                    tenv->ExceptionDescribe();
                    tenv->ExceptionClear();
                    ctx->cancel.store(true);   // lambda threw -> stop generating
                }
            });
        }
    };

    try {
        // Serialize against init/close. The runner itself is single-threaded
        // (HTP context is not safe to share across threads).
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->runner->ResetState();

        // Tokenize on this thread, then run prefill + decode loop.
        const auto ids = ctx->tokenizer->Encode(prompt_str, /*add_bos=*/true);
        if (!ctx->runner->Prefill(ids)) {
            LOGE("Prefill failed");
            if (lambdaGlobal) env->DeleteGlobalRef(lambdaGlobal);
            return env->NewStringUTF("");
        }

        // Wire sampling params for this call.
        gemma4::SamplingParams sp;
        if (temperature <= 0.0f) {
            sp.mode = gemma4::SamplingParams::Mode::kGreedy;
        } else {
            sp.mode        = gemma4::SamplingParams::Mode::kTemperature;
            sp.temperature = static_cast<float>(temperature);
        }
        sp.seed = static_cast<uint32_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());

        const int eos_id = ctx->tokenizer->EosId();
        const int limit  = std::max(1, static_cast<int>(maxTokens));

        std::vector<int32_t> partial_ids;
        partial_ids.reserve(limit);
        std::string last_decoded;

        for (int step = 0; step < limit; ++step) {
            if (ctx->cancel.load()) {
                LOGW("nativeGenerate: cancel requested at step %d", step);
                break;
            }

            const int32_t tok = ctx->runner->DecodeStep();
            if (tok < 0) {
                LOGE("DecodeStep returned %d", tok);
                break;
            }
            if (tok == eos_id) break;

            partial_ids.push_back(tok);

            // Incremental detokenization: decode the full id list and emit
            // the suffix delta. Cheap for SentencePiece, avoids partial-byte
            // splits on multi-byte UTF-8.
            const std::string whole = ctx->tokenizer->Decode(partial_ids);
            if (whole.size() > last_decoded.size() &&
                whole.compare(0, last_decoded.size(), last_decoded) == 0) {
                emit(whole.substr(last_decoded.size()));
                last_decoded = whole;
            }

            if (ctx->runner->IsFinished()) break;
        }
    } catch (const std::exception& e) {
        LOGE("nativeGenerate exception: %s", e.what());
    }

    if (lambdaGlobal) env->DeleteGlobalRef(lambdaGlobal);
    return env->NewStringUTF(accumulated.c_str());
}

// ---------------------------------------------------------------------------
// nativeCancel
// ---------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeCancel(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* ctx = CtxFromHandle(handle);
    if (!ctx) return;
    ctx->cancel.store(true);
    LOGI("nativeCancel: flag set");
}

// ---------------------------------------------------------------------------
// nativeReset — clear KV cache, keep QNN context loaded.
// ---------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeReset(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* ctx = CtxFromHandle(handle);
    if (!ctx) return;
    std::lock_guard<std::mutex> lk(ctx->mu);
    if (ctx->runner) ctx->runner->ResetState();
    ctx->cancel.store(false);
    LOGI("nativeReset ok");
}

// ---------------------------------------------------------------------------
// nativeClose
// ---------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeClose(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* ctx = CtxFromHandle(handle);
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->cancel.store(true);
        ctx->ready.store(false);
        ctx->runner.reset();
        ctx->tokenizer.reset();
        ctx->ple.reset();
    }
    LOGI("nativeClose handle=%p", static_cast<void*>(ctx));
    delete ctx;
}

}  // extern "C"
