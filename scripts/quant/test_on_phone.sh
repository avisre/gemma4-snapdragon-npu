#!/bin/bash
# Test a quantized .bin on the S22 (v69) via qnn-net-run.
# Usage: ./test_on_phone.sh <local_bin_path> <label>
set -e
BIN=$1
LABEL=$2
DEVICE=a5523839  # Galaxy S22

if [ -z "$BIN" ] || [ ! -f "$BIN" ]; then
    echo "Usage: $0 <bin_path> <label>"; exit 1
fi

SIZE_MB=$(du -m "$BIN" | cut -f1)
echo "==> Testing $LABEL ($SIZE_MB MB)"

REMOTE_DIR=/data/local/tmp/gemma4_quant_test
adb -s $DEVICE shell "mkdir -p $REMOTE_DIR"

echo "Pushing .bin..."
adb -s $DEVICE push "$BIN" $REMOTE_DIR/test.bin

# Build minimal input/output lists
echo "input_ids:1,32" > /tmp/input_list.txt   # we will craft a real input below
adb -s $DEVICE push /tmp/input_list.txt $REMOTE_DIR/

# Reuse the existing qnn-net-run + libs from /data/local/tmp/gemma4_v69
adb -s $DEVICE shell "cd $REMOTE_DIR && \
    cp /data/local/tmp/gemma4_v69/qnn-net-run ./ && \
    cp /data/local/tmp/gemma4_v69/lib*.so ./ && \
    cp /data/local/tmp/gemma4_v69/htp_config.json ./"

# Just try to load the binary via qnn-net-run --retrieve_context (no run, just load)
echo "Running qnn-net-run load test..."
START=$(date +%s.%N)
LOAD_LOG=$(adb -s $DEVICE shell "cd $REMOTE_DIR && \
    LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context test.bin \
    --config_file htp_config.json \
    --profiling_level basic 2>&1" || echo "QNN_FAILED")
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)
echo "Load took ${ELAPSED}s"
echo "=== qnn-net-run output ==="
echo "$LOAD_LOG" | head -40
echo "=== end ==="

if echo "$LOAD_LOG" | grep -q "QNN_FAILED\|Error\|error\|ERROR\|FAILED"; then
    echo "LOAD: FAIL"
    exit 2
else
    echo "LOAD: OK"
fi
