#!/bin/bash
# Captures native Mogwai references for Falcor/tests/image_tests scripts into
# tests/oracle/out-native/image-tests/<dir>/<name>/, as run_image_tests.py does: a generate
# script sets m.frameCapture.outputDir and m.script()s the test from its own directory.
# Usage: scripts/capture-native-image-tests.sh renderpasses/GBufferRT scene/Volumes ...
#        (no arguments: every script except DLSS/OptiX, which need NVIDIA SDKs)
W=$(cd "$(dirname "$0")/.." && pwd)
MOGWAI=${MOGWAI:-$W/Falcor/build/linux-gcc/bin/Release/Mogwai}
[ $# -eq 0 ] && set -- $(ls $W/Falcor/tests/image_tests/*/test_*.py | sed "s|$W/Falcor/tests/image_tests/||; s|/test_|/|; s|\.py||" | grep -v -E "DLSSPass|OptixDenoiser")
for t in "$@"; do
    dir=${t%/*}; name=${t#*/}
    out=$W/tests/oracle/out-native/image-tests/$dir/$name
    mkdir -p "$out"
    gen=$(mktemp --suffix=.py)
    printf 'm.frameCapture.outputDir = r"%s"\nm.script(r"%s")\n' "$out" "$W/Falcor/tests/image_tests/$dir/test_$name.py" > "$gen"
    (cd "$W/Falcor/tests/image_tests/$dir" && LD_LIBRARY_PATH=${FALCOR_LIBS:-$HOME/.conda/envs/falcor/lib} timeout 900 xvfb-run -a "$MOGWAI" --script "$gen" --headless --precise --logfile "$out/log.txt" > /dev/null 2>&1)
    echo "$t exit=$?"
    rm -f "$gen" "$out/log.txt"
    rmdir "$out" 2>/dev/null # nothing captured (e.g. GBufferRaster needs ROVs)
done
