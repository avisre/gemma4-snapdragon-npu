// gemma4_jni.h
//
// JNI function declarations for the Localyze.ai Gemma 4 E2B Android bridge
// (Kotlin Gemma4Runtime <-> native QNN HTP runner on Hexagon v69).
//
// All entry points follow the JNI name-mangling for the Kotlin class
// `ai.localyze.runtime.Gemma4Runtime`. The Kotlin side holds an opaque
// `long nativeHandle` that is created in `nativeCreate` and freed in
// `nativeClose`. Every other call takes that handle as its first jlong arg.
//
// Threading model
// ---------------
// The HTP backend is single-threaded per QNN context, so every native call
// for a given handle is serialized on a `RunnerContext`-owned worker
// thread. `nativeGenerate` blocks the caller until generation finishes,
// streaming each decoded token piece back through the supplied Kotlin
// `(String) -> Unit` lambda (kotlin.jvm.functions.Function1).
//
// Error model
// -----------
// - Boolean returns: JNI_FALSE on failure (errors go to logcat tag
//   "Gemma4JNI"; an exception is NOT thrown across the boundary).
// - Long  returns: 0L is sentinel for "create failed".
// - Int   returns: -1 indicates failure.
// - generate(): returns the accumulated string ("" on failure).
//
// ARM64 / NDK r26+ / C++17.

#ifndef GEMMA4_RUNTIME_ANDROID_GEMMA4_JNI_H_
#define GEMMA4_RUNTIME_ANDROID_GEMMA4_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Allocate the native RunnerContext and return an opaque handle.
// Signature: ()J
JNIEXPORT jlong JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeCreate(
        JNIEnv* env, jobject thiz);

// Load the QNN HTP backend (`libQnnHtp.so` via dlopen), deserialize the
// cached graph from `binPath`, mmap the PLE table from `pleBinPath`, and
// load the SentencePiece tokenizer from `tokenizerPath`.
//
// Returns JNI_TRUE on success, JNI_FALSE on any error.
// Signature: (JLjava/lang/String;Ljava/lang/String;Ljava/lang/String;)Z
JNIEXPORT jboolean JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeInit(
        JNIEnv* env, jobject thiz,
        jlong   handle,
        jstring binPath,
        jstring pleBinPath,
        jstring tokenizerPath);

// Free the runner, KV cache, PLE mmap, and QNN context. Idempotent on the
// native side; the Kotlin wrapper guarantees a single call.
// Signature: (J)V
JNIEXPORT void JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeClose(
        JNIEnv* env, jobject thiz, jlong handle);

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

// Run prefill on `prompt`, then autoregressively decode up to `maxTokens`
// new tokens. Each decoded piece is streamed back to `onToken`
// (kotlin.jvm.functions.Function1) on the inference worker thread. Blocking.
//
// Returns the full accumulated generation as a Java UTF-8 string.
// Signature: (JLjava/lang/String;IFLkotlin/jvm/functions/Function1;)Ljava/lang/String;
JNIEXPORT jstring JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeGenerate(
        JNIEnv* env, jobject thiz,
        jlong   handle,
        jstring prompt,
        jint    maxTokens,
        jfloat  temperature,
        jobject onTokenLambda);

// Best-effort cooperative cancel of an in-flight nativeGenerate. The decode
// loop checks the cancel flag between every token and returns early. Safe
// to call from any thread.
// Signature: (J)V
JNIEXPORT void JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeCancel(
        JNIEnv* env, jobject thiz, jlong handle);

// Reset the KV cache and conversation state without tearing down QNN.
// Useful between independent chats. Signature: (J)V
JNIEXPORT void JNICALL
Java_ai_localyze_runtime_Gemma4Runtime_nativeReset(
        JNIEnv* env, jobject thiz, jlong handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GEMMA4_RUNTIME_ANDROID_GEMMA4_JNI_H_
