#!/bin/bash
# Download QNN runtime 2.46 libs from Maven (66 MB), extract for Android arm64.
# These are required to load AI Hub-compiled .bin files on the phone.
# AI Hub uses QAIRT 2.45; runtime 2.46 reads 2.45 binaries.
set -euo pipefail

VERSION="${QNN_VERSION:-2.46.0}"
DEST="${1:-./qnn_runtime}"

mkdir -p "$DEST"
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "==> Downloading QNN runtime $VERSION ($66 MB)..."
wget -q "https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/${VERSION}/qnn-runtime-${VERSION}.aar" -O "$TMP/qnn.aar"

echo "==> Extracting Android arm64 libs..."
unzip -q "$TMP/qnn.aar" "jni/arm64-v8a/*" -d "$TMP"

cp "$TMP/jni/arm64-v8a/"libQnn*.so "$DEST/"

echo "==> Installed to $DEST:"
ls -la "$DEST/"libQnn*.so | awk '{printf "    %s (%s bytes)\n", $NF, $5}'

echo ""
echo "Next: adb push $DEST/libQnn*.so /data/local/tmp/gemma4_v69/"
