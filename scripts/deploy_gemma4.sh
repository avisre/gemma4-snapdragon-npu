#!/usr/bin/env bash
# deploy_gemma4.sh - Deploy Gemma 4 E2B (Hexagon v69) to a OnePlus 10 Pro.
#
# What this script does:
#   1. Confirms a v69-capable device is connected (taro / kalama / OnePlus 10 Pro).
#   2. Verifies all required local artifacts exist (.pte, tokenizer, PLE blob,
#      QNN runtime libs, gemma4_runner binary).
#   3. Pushes everything to /data/local/tmp/gemma4_v69/ on device.
#   4. Runs a one-prompt smoke test through the custom gemma4_runner.
#
# Important: Gemma 4 E2B is NOT a vanilla decoder. The stock qnn_llama_runner
# from the Qwen3 zip supports {llama2, llama3, qwen2_5, qwen3, gemma3, phi4,
# smollm2_135m, smollm3} — there is NO `gemma4` decoder version. We use a
# custom binary at $RUNNER_PATH that knows how to feed per_layer_inputs from
# the PLE preprocessor; we reuse only the QNN .so libs from the Qwen3 zip.
#
# Usage:
#   bash runtime/deploy_gemma4.sh [PTE_PATH]
# Env overrides:
#   DEVICE_SERIAL    adb serial (default: first 'device'-state serial)
#   TOKENIZER_PATH   default: ../checkpoints/gemma-4-e2b-it/tokenizer.json
#   PLE_PATH         default: ../aihub/gemma4_v69.ple
#   RUNNER_PATH      default: ./build/gemma4_runner
#   QWEN3_SRC        default: /home/hardoker77/v69_llm/qwen3_v69
#   PROMPT           default: "The capital of France is"
#   SEQ_LEN          default: 128
#   SKIP_RUN=1       push everything but do not invoke the runner

set -euo pipefail

# ---------- Paths ----------
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

PTE_PATH="${1:-${PTE_PATH:-$PROJECT_ROOT/executorch/build-x86/gemma4_e2b/hybrid_llama_qnn.pte}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJECT_ROOT/checkpoints/gemma-4-e2b-it/tokenizer.json}"
PLE_PATH="${PLE_PATH:-$PROJECT_ROOT/aihub/gemma4_v69.ple}"
RUNNER_PATH="${RUNNER_PATH:-$HERE/build/gemma4_runner}"
QWEN3_SRC="${QWEN3_SRC:-/home/hardoker77/v69_llm/qwen3_v69}"
QWEN3_ZIP="$QWEN3_SRC/qnn_llama_runner.zip"

PHONE_DIR="/data/local/tmp/gemma4_v69"
PROMPT="${PROMPT:-The capital of France is}"
SEQ_LEN="${SEQ_LEN:-128}"

LOG_DIR="$HERE/phone_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deploy_$(date +%Y%m%d_%H%M%S).log"

log() { printf '[deploy] %s\n' "$*" | tee -a "$LOG_FILE"; }
die() { printf '[deploy][FAIL] %s\n' "$*" | tee -a "$LOG_FILE" >&2; exit 1; }

trap 'rm -rf "${TMPLIB:-}"' EXIT

# ---------- Step 1: ADB + device class ----------
log "==> Step 1: verifying ADB target"
command -v adb >/dev/null 2>&1 || die "adb not on PATH"
command -v unzip >/dev/null 2>&1 || die "unzip not on PATH"

DEVICES="$(adb devices | awk 'NR>1 && $2=="device" {print $1}')"
[ -n "$DEVICES" ] || die "no adb devices in 'device' state (plug in the OnePlus 10 Pro and run 'adb devices')"

if [ -z "${DEVICE_SERIAL:-}" ]; then
    DEVICE_SERIAL="$(echo "$DEVICES" | head -1)"
fi
echo "$DEVICES" | grep -qx "$DEVICE_SERIAL" || die "DEVICE_SERIAL=$DEVICE_SERIAL not in 'device' list: $DEVICES"
ADB="adb -s $DEVICE_SERIAL"

MODEL="$($ADB shell getprop ro.product.model       | tr -d '\r')"
PLATFORM="$($ADB shell getprop ro.board.platform   | tr -d '\r')"
SOC_MODEL="$($ADB shell getprop ro.soc.model       | tr -d '\r' 2>/dev/null || true)"
log "    serial=$DEVICE_SERIAL  model=$MODEL  platform=$PLATFORM  soc=$SOC_MODEL"

# Hexagon v69 chips: SM8450 (taro, "Snapdragon 8 Gen 1"). Hexagon v73: SM8550 (kalama).
# Anything else (e.g. sdm845/845/OnePlus 6 -> Hexagon v65) WILL crash on our .pte.
case "$PLATFORM" in
    taro|kalama|pineapple)
        log "    platform OK ($PLATFORM has Hexagon v69+)" ;;
    sdm845|sm8150|kona|lahaina)
        die "Connected device is platform '$PLATFORM' ($MODEL).
       This NPU is older than Hexagon v69 and CANNOT run the gemma4_e2b.pte we produce.
       Please unplug this device and connect the OnePlus 10 Pro (NE2213/NE2215/NE2210, platform=taro)." ;;
    *)
        die "Connected device is platform '$PLATFORM' ($MODEL), expected 'taro' (Snapdragon 8 Gen 1, Hexagon v69).
       Please plug in the OnePlus 10 Pro." ;;
esac

# ---------- Step 2: validate local artifacts ----------
log "==> Step 2: validating local artifacts"
for pair in "PTE:$PTE_PATH" "TOKENIZER:$TOKENIZER_PATH" "PLE:$PLE_PATH" "RUNNER:$RUNNER_PATH" "QWEN3_ZIP:$QWEN3_ZIP"; do
    name="${pair%%:*}"; path="${pair#*:}"
    if [ ! -f "$path" ]; then
        die "$name not found at $path
       (override with ${name}_PATH env var, or pass PTE_PATH as positional arg)"
    fi
    size_h="$(du -h "$path" | cut -f1)"
    log "    $name = $path  ($size_h)"
done

# ---------- Step 3: stage Qwen3 QNN libs ----------
log "==> Step 3: staging QNN runtime libs from Qwen3 zip"
TMPLIB="$(mktemp -d)"
unzip -o -q "$QWEN3_ZIP" -d "$TMPLIB"

# Only need the v69 stub+skel + the common libs. v73/v75/v79 are skipped to save bandwidth;
# add them back if you also test on later devices from the same script.
QNN_LIB_DIR="$TMPLIB/qnn_llama_runner"
for lib in libQnnHtp.so libQnnSystem.so libQnnHtpV69Stub.so libQnnHtpV69Skel.so libqnn_executorch_backend.so; do
    [ -f "$QNN_LIB_DIR/$lib" ] || die "expected $lib not found in $QWEN3_ZIP"
done
log "    QNN libs ready in $QNN_LIB_DIR"

# ---------- Step 4: push everything ----------
log "==> Step 4: pushing to $PHONE_DIR"
$ADB shell "mkdir -p $PHONE_DIR" >>"$LOG_FILE" 2>&1

# QNN .so libs (from Qwen3 zip).
for lib in libQnnHtp.so libQnnSystem.so libQnnHtpV69Stub.so libQnnHtpV69Skel.so libqnn_executorch_backend.so; do
    $ADB push "$QNN_LIB_DIR/$lib" "$PHONE_DIR/$lib" >>"$LOG_FILE" 2>&1
done

# Gemma 4 artifacts (NOT Qwen's).
$ADB push "$PTE_PATH"       "$PHONE_DIR/gemma4_e2b.pte"  >>"$LOG_FILE" 2>&1
$ADB push "$TOKENIZER_PATH" "$PHONE_DIR/tokenizer.json"  >>"$LOG_FILE" 2>&1
$ADB push "$PLE_PATH"       "$PHONE_DIR/gemma4_v69.ple"  >>"$LOG_FILE" 2>&1
$ADB push "$RUNNER_PATH"    "$PHONE_DIR/gemma4_runner"   >>"$LOG_FILE" 2>&1
$ADB shell "chmod 755 $PHONE_DIR/gemma4_runner"          >>"$LOG_FILE" 2>&1

log "    push complete; contents:"
$ADB shell "ls -la $PHONE_DIR" | tee -a "$LOG_FILE"

if [ "${SKIP_RUN:-0}" = "1" ]; then
    log "==> SKIP_RUN=1, not invoking runner. Files staged."
    exit 0
fi

# ---------- Step 5: smoke test ----------
log "==> Step 5: smoke test (prompt: \"$PROMPT\", seq_len=$SEQ_LEN)"
RAW_OUT="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

$ADB shell <<ADBEOF 2>&1 | tee "$RAW_OUT" >>"$LOG_FILE"
set +e
cd $PHONE_DIR
export LD_LIBRARY_PATH=\$PWD
export ADSP_LIBRARY_PATH=\$PWD:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp
./gemma4_runner \\
    --model_path     gemma4_e2b.pte \\
    --tokenizer_path tokenizer.json \\
    --ple_path       gemma4_v69.ple \\
    --prompt         "$PROMPT" \\
    --seq_len        $SEQ_LEN \\
    --kv_updater     SmartMask \\
    --eval_mode      1 \\
    --temperature    0.0 \\
    --report_timing  1
RC=\$?
echo "--- outputs.txt ---"
[ -f outputs.txt ] && cat outputs.txt || echo "(no outputs.txt)"
exit \$RC
ADBEOF

log "==> Deploy + smoke test complete."
log "    raw output: $RAW_OUT"
log "    full log:   $LOG_FILE"
