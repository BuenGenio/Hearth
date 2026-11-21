#!/bin/sh
# Download the models Hearth needs into a directory (default ./models).
# The resulting files are checked against pinned SHA-256 sums; a mismatch aborts,
# so a changed upstream file is noticed.
set -eu

DEST="${1:-models}"

# Pinned checksums of the files as used (verified at the end).
PINS="4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf  yamnet.tflite
9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6  silero_vad.onnx
c8a2a0129c4ab8e463164c142f82d25649661b122c8cd0b7aab5c9e80b90ad24  kws/bpe.model
f61ebd3eed3773a44d088d53dfae92dbb6aec4839f4dcaee2d402414741663a3  kws/decoder-epoch-12-avg-2-chunk-16-left-64.onnx
063fbc1aeae8a9b574607a331a00e60371846ef9eaa3c1d9ea48176665dfc693  kws/encoder-epoch-12-avg-2-chunk-16-left-64.onnx
0d7a37e749d8055223029318d6ffae82db1dae2d315d0892a68ba5dad17c1d2d  kws/joiner-epoch-12-avg-2-chunk-16-left-64.onnx
fd2ded4050a55d2b1578870ba8697d02371980217806b7558bd0a5cc60f3ba53  kws/tokens.txt"
mkdir -p "$DEST/kws"

fetch() { # url dest sha256
    if [ -f "$2" ]; then        # integrity is checked for every file at the end
        return 0
    fi
    echo "fetching $(basename "$2")"
    curl -fsSL --retry 3 -o "$2.part" "$1"
    if [ "$3" != "-" ]; then
        echo "$3  $2.part" | sha256sum -c - >/dev/null || {
            echo "checksum mismatch for $1: got $(sha256sum "$2.part" | cut -d' ' -f1)" >&2
            exit 1
        }
    fi
    mv "$2.part" "$2"
}

# YAMNet (AudioSet, 521 classes), TFLite build hosted with MediaPipe. Labels are
# embedded in the file as a zip member (yamnet_label_list.txt).
fetch "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite" \
      "$DEST/yamnet.tflite" "${YAMNET_SHA256:--}"

# Silero VAD, ONNX export used by sherpa-onnx.
fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
      "$DEST/silero_vad.onnx" "${SILERO_SHA256:--}"

# Open-vocabulary keyword spotter (English, GigaSpeech, 3.3 M parameters).
KWS=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
if [ ! -f "$DEST/kws/tokens.txt" ]; then
    fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$KWS.tar.bz2" \
          "$DEST/$KWS.tar.bz2" "${KWS_SHA256:--}"
    tar -xjf "$DEST/$KWS.tar.bz2" -C "$DEST"
    for f in tokens.txt bpe.model \
             encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
             decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
             joiner-epoch-12-avg-2-chunk-16-left-64.onnx; do
        cp "$DEST/$KWS/$f" "$DEST/kws/$f"
    done
    rm -rf "$DEST/$KWS" "$DEST/$KWS.tar.bz2"
fi

(cd "$DEST" && echo "$PINS" | sha256sum -c --quiet -) || { echo "model checksum mismatch" >&2; exit 1; }
echo "models ok in $DEST"
