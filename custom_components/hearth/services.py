"""Service actions for Hearth.

Registered once from async_setup, so they stay available (and give a clear
error) when the hub's config entry is missing or not loaded.
"""

from collections.abc import Awaitable, Callable
from typing import Any

import probatio

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_CONFIG_ENTRY_ID
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv, service
from homeassistant.helpers.selector import ConfigEntrySelector
from homeassistant.util import dt as dt_util

from .api import HearthApiError, HearthAuthError, HearthClient, HearthConnectionError
from .const import (
    ATTR_END,
    ATTR_HOURS,
    ATTR_KIND,
    ATTR_LIMIT,
    ATTR_QUERY,
    ATTR_START,
    DOMAIN,
    SERVICE_GET_TRANSCRIPT,
    SERVICE_LATEST_SUMMARY,
    SERVICE_SEARCH,
    SERVICE_SUMMARISE,
)

# Optional: with a single hub configured, the action finds it by itself.
_ENTRY: dict[Any, Any] = {
    probatio.Optional(ATTR_CONFIG_ENTRY_ID): ConfigEntrySelector({"integration": DOMAIN})
}
_KIND = probatio.All(cv.string, probatio.Length(min=1, max=32))

# Limits mirror the hub's own (docs/openapi.json).
SEARCH_SCHEMA = probatio.Schema(
    {
        **_ENTRY,
        probatio.Required(ATTR_QUERY): probatio.All(
            cv.string, probatio.Length(min=1, max=200)
        ),
        probatio.Optional(ATTR_LIMIT, default=20): probatio.All(
            probatio.Coerce(int), probatio.Range(min=1, max=500)
        ),
    }
)
GET_TRANSCRIPT_SCHEMA = probatio.Schema(
    {
        **_ENTRY,
        probatio.Required(ATTR_START): cv.datetime,
        probatio.Optional(ATTR_END): cv.datetime,
        probatio.Optional(ATTR_LIMIT, default=1000): probatio.All(
            probatio.Coerce(int), probatio.Range(min=1, max=10000)
        ),
    }
)
SUMMARISE_SCHEMA = probatio.Schema(
    {
        **_ENTRY,
        probatio.Optional(ATTR_HOURS, default=1): probatio.All(
            probatio.Coerce(float), probatio.Range(min=0, min_included=False, max=72)
        ),
        probatio.Optional(ATTR_KIND, default="manual"): _KIND,
    }
)
LATEST_SUMMARY_SCHEMA = probatio.Schema(
    {
        **_ENTRY,
        probatio.Optional(ATTR_KIND): _KIND,
    }
)


async def _async_hub_call[T](
    call: ServiceCall, func: Callable[[HearthClient], Awaitable[T]]
) -> T:
    """Find the loaded hub at call time, run func, turn hub errors into HA errors."""
    entry: ConfigEntry[HearthClient] = service.async_get_config_entry(
        call.hass, DOMAIN, call.data.get(ATTR_CONFIG_ENTRY_ID)
    )
    client = entry.runtime_data
    try:
        return await func(client)
    except HearthAuthError as err:
        entry.async_start_reauth(call.hass)
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="auth_failed",
            translation_placeholders={"title": entry.title},
        ) from err
    except HearthConnectionError as err:
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="cannot_connect",
            translation_placeholders={"url": client.url, "error": str(err)},
        ) from err
    except HearthApiError as err:
        if err.status in (400, 422):  # the hub did not like what was asked
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="request_rejected",
                translation_placeholders={"detail": err.detail},
            ) from err
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="api_error",
            translation_placeholders={"status": str(err.status), "detail": err.detail},
        ) from err


async def _async_search(call: ServiceCall) -> ServiceResponse:
    """Full-text search over the transcripts."""
    segments = await _async_hub_call(
        call, lambda hub: hub.search(call.data[ATTR_QUERY], call.data[ATTR_LIMIT])
    )
    return {"segments": segments}


async def _async_get_transcript(call: ServiceCall) -> ServiceResponse:
    """Transcripts in a time range. Times without an offset are local time."""
    start = dt_util.as_utc(call.data[ATTR_START])
    end = dt_util.as_utc(call.data[ATTR_END]) if ATTR_END in call.data else dt_util.utcnow()
    if end <= start:
        raise ServiceValidationError(
            translation_domain=DOMAIN, translation_key="end_before_start"
        )
    segments = await _async_hub_call(
        call, lambda hub: hub.segments(start, end, call.data[ATTR_LIMIT])
    )
    return {"segments": segments}


async def _async_summarise(call: ServiceCall) -> ServiceResponse:
    """Summarise the last hours now; the hub stores and publishes the result."""
    summary = await _async_hub_call(
        call, lambda hub: hub.run_summary(call.data[ATTR_HOURS], call.data[ATTR_KIND])
    )
    return {"summary": summary} if call.return_response else None


async def _async_latest_summary(call: ServiceCall) -> ServiceResponse:
    """The newest stored summary, optionally of one kind."""
    summary = await _async_hub_call(
        call, lambda hub: hub.latest_summary(call.data.get(ATTR_KIND))
    )
    return {"summary": summary}


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register the Hearth service actions."""
    hass.services.async_register(
        DOMAIN,
        SERVICE_SEARCH,
        _async_search,
        schema=SEARCH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_TRANSCRIPT,
        _async_get_transcript,
        schema=GET_TRANSCRIPT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SUMMARISE,
        _async_summarise,
        schema=SUMMARISE_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_LATEST_SUMMARY,
        _async_latest_summary,
        schema=LATEST_SUMMARY_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
