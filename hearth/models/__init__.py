"""Model runners. Each wraps one inference runtime behind a small interface and
imports its native library lazily, so the rest of Hearth (and its tests) loads
without them. No Home Assistant, MQTT or SQLite imports here.
"""
