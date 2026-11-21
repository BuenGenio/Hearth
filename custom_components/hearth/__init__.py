"""The Hearth integration.

Hearth's entities reach Home Assistant through MQTT discovery. This integration
adds what discovery cannot: service actions backed by the hub's HTTP API. It
creates no entities and no devices.
"""

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_TOKEN, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.typing import ConfigType

from .api import HearthAuthError, HearthClient, HearthError
from .const import DOMAIN
from .services import async_setup_services

type HearthConfigEntry = ConfigEntry[HearthClient]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the service actions; they exist whether or not a hub is loaded."""
    async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: HearthConfigEntry) -> bool:
    """Set up one hub: check that it answers and that the token still works."""
    client = HearthClient(
        async_get_clientsession(hass), entry.data[CONF_URL], entry.data[CONF_API_TOKEN]
    )
    try:
        await client.status()
    except HearthAuthError as err:
        raise ConfigEntryAuthFailed(
            translation_domain=DOMAIN,
            translation_key="auth_failed",
            translation_placeholders={"title": entry.title},
        ) from err
    except HearthError as err:
        raise ConfigEntryNotReady(
            translation_domain=DOMAIN,
            translation_key="cannot_connect",
            translation_placeholders={"url": client.url, "error": str(err)},
        ) from err
    entry.runtime_data = client
    return True


async def async_unload_entry(hass: HomeAssistant, entry: HearthConfigEntry) -> bool:
    """Unload a hub. Nothing to tear down: the session belongs to Home Assistant."""
    return True
