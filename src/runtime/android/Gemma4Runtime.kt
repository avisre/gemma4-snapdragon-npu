// Gemma4Runtime.kt
//
// Kotlin facade for the Gemma 4 E2B on-device runner (Snapdragon Hexagon v69
// via QNN HTP). Backed by libgemma4_jni.so; see android/gemma4_jni.cpp.
//
// Mirrors the existing Localyze.ai LLM-runtime contract: caller supplies
// model artifact paths in the constructor, calls `generate(...)` from a
// worker thread (it blocks while streaming tokens to the callback), and
// disposes with `close()`.
//
// IMPORTANT
// ---------
// * All three paths must already exist on-device (under filesDir, typically
//   `/data/data/<pkg>/files/gemma4/`). The class does not download.
// * QNN HTP is single-context-per-process in practice — keep at most one
//   `Gemma4Runtime` instance alive at a time.
// * `generate()` blocks. Run it on Dispatchers.Default / IO.
// * The `onToken` lambda fires on the inference worker thread; if you need
//   to update UI, marshal to the main dispatcher yourself.
// * Snapdragon-only. Caller is responsible for refusing to construct on
//   incompatible hardware (matches the project rule: NPU/GPU only, never
//   CPU backend on Android).

package ai.localyze.runtime

import androidx.annotation.AnyThread
import androidx.annotation.WorkerThread
import java.util.concurrent.atomic.AtomicBoolean

/**
 * On-device Gemma 4 E2B runtime.
 *
 * @param binPath        absolute path to the compiled QNN context binary
 *                       (e.g. `.../gemma4/gemma4_e2b_v69.bin`).
 * @param pleBinPath     absolute path to the externalized Per-Layer
 *                       Embedding table (`.../gemma4/ple_weights.bin`).
 * @param tokenizerPath  absolute path to the Gemma SentencePiece tokenizer
 *                       (`.../gemma4/tokenizer.json`).
 *
 * Construction is cheap (allocates an opaque native handle). The expensive
 * model load happens lazily on the first `generate()` call, or eagerly if
 * the caller invokes [warmUp].
 */
class Gemma4Runtime(
    private val binPath: String,
    private val pleBinPath: String,
    private val tokenizerPath: String,
) : AutoCloseable {

    companion object {
        private const val TAG = "Gemma4Runtime"

        init {
            // libgemma4_jni.so — packaged in jniLibs/arm64-v8a/.
            // Sibling libQnnHtp.so / libQnnSystem.so must also be present;
            // they are dlopen'd from the native side.
            System.loadLibrary("gemma4_jni")
        }
    }

    // Opaque pointer to the native RunnerContext.
    private var nativeHandle: Long = 0L
    private val closed = AtomicBoolean(false)
    private val initialized = AtomicBoolean(false)

    init {
        nativeHandle = nativeCreate()
        check(nativeHandle != 0L) { "Gemma4Runtime: nativeCreate returned null" }
    }

    /**
     * Eagerly load the QNN context, PLE table, and tokenizer. Optional —
     * `generate()` will lazy-init on first call. Returns true on success.
     * Blocking; expect ~1-3 s on first cold load.
     */
    @WorkerThread
    fun warmUp(): Boolean {
        ensureOpen()
        return ensureInitialized()
    }

    /**
     * Run prefill + decode for [prompt]. Blocks the calling thread, streaming
     * each decoded text piece to [onToken] on the inference worker thread.
     *
     * Returns the full accumulated generation (also useful for non-streaming
     * callers — pass a no-op lambda).
     *
     * @param prompt       the full prompt text (chat template already applied
     *                     by the caller).
     * @param maxTokens    upper bound on new tokens (EOS stops earlier).
     * @param temperature  0f or negative -> greedy; otherwise temperature
     *                     sampling. Recommended range: 0.0–1.0.
     * @param onToken      callback fired for every decoded text piece.
     */
    @WorkerThread
    fun generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7f,
        onToken: (String) -> Unit,
    ): String {
        ensureOpen()
        require(maxTokens > 0) { "maxTokens must be > 0" }
        if (!ensureInitialized()) {
            throw IllegalStateException("Gemma4Runtime failed to initialize")
        }
        return nativeGenerate(nativeHandle, prompt, maxTokens, temperature, onToken)
    }

    /**
     * Best-effort cooperative cancel of an in-flight [generate]. Safe to call
     * from any thread; the decode loop checks between every token.
     */
    @AnyThread
    fun cancel() {
        if (closed.get() || nativeHandle == 0L) return
        nativeCancel(nativeHandle)
    }

    /**
     * Clear KV cache and conversation state without unloading the model.
     * Use between independent chats.
     */
    @WorkerThread
    fun reset() {
        ensureOpen()
        nativeReset(nativeHandle)
    }

    /** Release native resources. Idempotent. */
    @AnyThread
    override fun close() {
        if (closed.compareAndSet(false, true)) {
            val h = nativeHandle
            nativeHandle = 0L
            if (h != 0L) nativeClose(h)
        }
    }

    // ------------------------------------------------------------------ //
    // Internals
    // ------------------------------------------------------------------ //

    private fun ensureOpen() {
        check(!closed.get() && nativeHandle != 0L) {
            "Gemma4Runtime is closed"
        }
    }

    @Synchronized
    private fun ensureInitialized(): Boolean {
        if (initialized.get()) return true
        val ok = nativeInit(nativeHandle, binPath, pleBinPath, tokenizerPath)
        if (ok) initialized.set(true)
        return ok
    }

    // ------------------------------------------------------------------ //
    // JNI declarations — implemented in android/gemma4_jni.cpp.
    // ------------------------------------------------------------------ //
    private external fun nativeCreate(): Long
    private external fun nativeInit(
        handle: Long,
        binPath: String,
        pleBinPath: String,
        tokenizerPath: String,
    ): Boolean
    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        onToken: (String) -> Unit,
    ): String
    private external fun nativeCancel(handle: Long)
    private external fun nativeReset(handle: Long)
    private external fun nativeClose(handle: Long)
}
