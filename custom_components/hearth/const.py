"""Constants for the Hearth integration."""

from typing import Final

DOMAIN: Final = "hearth"

DEFAULT_URL: Final = "http://homeassistant-host.local:8377"

# Seconds. Summaries call an LLM on the hub synchronously, so they get far longer.
REQUEST_TIMEOUT: Final = 15.0
SUMMARY_TIMEOUT: Final = 330.0

SERVICE_SEARCH: Final = "search"
SERVICE_GET_TRANSCRIPT: Final = "get_transcript"
SERVICE_SUMMARISE: Final = "summarise"
SERVICE_LATEST_SUMMARY: Final = "latest_summary"

ATTR_QUERY: Final = "query"
ATTR_LIMIT: Final = "limit"
ATTR_START: Final = "start"
ATTR_END: Final = "end"
ATTR_HOURS: Final = "hours"
ATTR_KIND: Final = "kind"
