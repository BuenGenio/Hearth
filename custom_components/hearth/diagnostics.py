"""Diagnostics for Hearth: the config entry and the hub's own status."""

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_API_TOKEN
from homeassistant.core import HomeAssistant

from . import HearthConfigEntry
from .api import HearthError

TO_REDACT = {CONF_API_TOKEN, "token", "tokens", "authorization", "password", "api_key"}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: HearthConfigEntry
) -> dict[str, Any]:
    """Return the hub's /api/v1/status, with the token redacted."""
    try:
        status: dict[str, Any] = await entry.runtime_data.status()
    except HearthError as err:
        status = {"error": str(err)}
    return {
        "entry": async_redact_data(entry.as_dict(), TO_REDACT),
        "status": async_redact_data(status, TO_REDACT),
    }
