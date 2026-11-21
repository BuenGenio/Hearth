"""Config flow for Hearth."""

from collections.abc import Mapping
import logging
from typing import Any

import probatio
from yarl import URL

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.const import CONF_API_TOKEN, CONF_URL
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .api import HearthAuthError, HearthClient, HearthError
from .const import DEFAULT_URL, DOMAIN

_LOGGER = logging.getLogger(__name__)

_TOKEN_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

STEP_USER_SCHEMA = probatio.Schema(
    {
        probatio.Required(CONF_URL, default=DEFAULT_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        probatio.Required(CONF_API_TOKEN): _TOKEN_SELECTOR,
    }
)
STEP_REAUTH_SCHEMA = probatio.Schema({probatio.Required(CONF_API_TOKEN): _TOKEN_SELECTOR})


def normalise_url(value: str) -> str | None:
    """Return the hub base URL without trailing slash, or None if unusable."""
    try:
        url = URL(value.strip())
    except (TypeError, ValueError):
        return None
    if url.scheme not in ("http", "https") or not url.host:
        return None
    return str(url.with_query(None).with_fragment(None)).rstrip("/")


class HearthConfigFlow(ConfigFlow, domain=DOMAIN):
    """Ask for the hub URL and an API token, and check them against the hub."""

    VERSION = 1

    async def _async_check(self, url: str, token: str) -> dict[str, str]:
        """Call GET /api/v1/status with the token; return form errors."""
        client = HearthClient(async_get_clientsession(self.hass), url, token)
        try:
            status = await client.status()
        except HearthAuthError:
            return {"base": "invalid_auth"}
        except HearthError:
            return {"base": "cannot_connect"}
        except Exception:
            _LOGGER.exception("Unexpected error while checking the Hearth hub")
            return {"base": "unknown"}
        if "version" not in status:  # something answered, but it is not a Hearth hub
            return {"base": "cannot_connect"}
        return {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the step started by the user."""
        errors: dict[str, str] = {}
        if user_input is not None:
            token = user_input[CONF_API_TOKEN].strip()
            if (url := normalise_url(user_input[CONF_URL])) is None:
                errors[CONF_URL] = "invalid_url"
            else:
                # The hub exposes no stable identifier and a URL is not an
                # acceptable unique ID, so duplicates are matched on the URL.
                self._async_abort_entries_match({CONF_URL: url})
                errors = await self._async_check(url, token)
            if not errors:
                return self.async_create_entry(
                    title=f"Hearth ({URL(url).host})",
                    data={CONF_URL: url, CONF_API_TOKEN: token},
                )
        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_SCHEMA, user_input
            ),
            errors=errors,
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle a rejected token."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a new token for the same hub."""
        errors: dict[str, str] = {}
        entry = self._get_reauth_entry()
        if user_input is not None:
            token = user_input[CONF_API_TOKEN].strip()
            errors = await self._async_check(entry.data[CONF_URL], token)
            if not errors:
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_API_TOKEN: token}
                )
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=STEP_REAUTH_SCHEMA,
            errors=errors,
            description_placeholders={"url": entry.data[CONF_URL]},
        )
