#!/bin/bash
# Watch for new decode .bin files, push to phone and smoke-test.
set -u

DEV=a5523839
LOCAL_DIR=/home/hardoker77/gemma4_e2b_v69/aihub

declare -A pushed
for i in 0 1 2 3 4; do
    pushed[prefill_$i]=1
done
# Already pushed prefills above; track decodes
for i in 0 1 2 3 4; do
    pushed[decode_$i]=0
done

while true; do
    all_done=1
    for i in 0 1 2 3 4; do
        key=decode_$i
        if [ "${pushed[$key]}" = "0" ]; then
            f=$LOCAL_DIR/gemma4_hybrid_decode_part$i.bin
            if [ -f "$f" ]; then
                # File exists locally. Check it's stable (not still being written)
                s1=$(stat -c%s "$f" 2>/dev/null || echo 0)
                sleep 5
                s2=$(stat -c%s "$f" 2>/dev/null || echo 0)
                if [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ]; then
                    echo "[$(date +%H:%M:%S)] decode_part$i ready ($s1 bytes). Pushing..."
                    if adb -s $DEV push "$f" /data/local/tmp/gemma4_v69/ 2>&1 | tail -1; then
                        echo "[$(date +%H:%M:%S)] decode_part$i pushed. Smoke testing..."
                        adb -s $DEV shell "cd /data/local/tmp/gemma4_v69 && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. timeout 50 ./qnn-net-run --backend libQnnHtp.so --retrieve_context gemma4_hybrid_decode_part$i.bin 2>&1 | tail -5"
                        pushed[$key]=1
                    fi
                else
                    all_done=0
                fi
            else
                all_done=0
            fi
        fi
    done
    if [ "$all_done" = "1" ]; then
        echo "[$(date +%H:%M:%S)] All 5 decode parts pushed+tested. Done."
        break
    fi
    sleep 20
done
