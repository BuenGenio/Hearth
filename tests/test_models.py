"""The real models on real audio. Skipped unless the models are downloaded."""
import pytest

from hearth.capture.sources import read_wav
from hearth.models.kws import SherpaKws, normalise
from hearth.models.vad import SherpaVad
from hearth.models.yamnet import WINDOW, Yamnet
from conftest import FIXTURES

pytestmark = pytest.mark.models
PHRASES = ["hearth start recording", "Hearth, stop recording."]


def feed_all(spotter, audio):
    import numpy as np

    hits = []
    audio = np.concatenate([np.zeros(8000, np.float32), audio, np.zeros(16000, np.float32)])
    for k in range(0, len(audio), 1600):
        hits += spotter.feed(audio[k : k + 1600])
    return hits


def test_yamnet_names_speech_and_knows_the_guard_classes(models_dir):
    from hearth.config import default_categories
    from hearth.guard import GuardLogic

    y = Yamnet(f"{models_dir}/yamnet.tflite")
    assert len(y.labels) == 521
    GuardLogic(default_categories(), y.labels)            # every configured class exists in the model
    scores = y.scores(read_wav(f"{FIXTURES}/tts_3.wav")[8000 : 8000 + WINDOW])
    assert y.labels[int(scores.argmax())] == "Speech"


def test_wake_phrases_are_recognised_and_plain_speech_is_not(models_dir):
    for wav, expected in (("tts_1.wav", ["HEARTH START RECORDING"]), ("tts_2.wav", ["HEARTH STOP RECORDING"]),
                          ("tts_3.wav", [])):
        spotter = SherpaKws(f"{models_dir}/kws", PHRASES)
        assert feed_all(spotter, read_wav(f"{FIXTURES}/{wav}")) == expected
    assert normalise("Hearth, stop recording.") == "HEARTH STOP RECORDING"


def test_vad_finds_the_sentence(models_dir):
    import numpy as np

    audio = read_wav(f"{FIXTURES}/tts_3.wav")
    vad = SherpaVad(f"{models_dir}/silero_vad.onnx")
    padded = np.concatenate([np.zeros(16000, np.float32), audio, np.zeros(32000, np.float32)])
    segs = []
    for k in range(0, len(padded), 1600):
        segs += vad.feed(padded[k : k + 1600])
    segs += vad.flush()
    assert len(segs) >= 1
    assert abs(segs[0][0] - 16000) < 8000                 # starts near where the speech starts
    assert sum(len(s) for _, s in segs) > 0.7 * len(audio)
