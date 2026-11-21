"""Diagnostics: hub status included, token redacted."""

import json

from aiohttp import ClientConnectionError
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.diagnostics import (
    get_diagnostics_for_config_entry,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker
from pytest_homeassistant_custom_component.typing import ClientSessionGenerator

from homeassistant.components.diagnostics import REDACTED
from homeassistant.const import CONF_API_TOKEN, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from custom_components.hearth.diagnostics import async_get_config_entry_diagnostics

from .conftest import HUB, STATUS, STATUS_URL, TOKEN


async def test_diagnostics(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Downloaded through Home Assistant's diagnostics endpoint."""
    assert await async_setup_component(hass, "diagnostics", {})
    await hass.async_block_till_done()

    result = await get_diagnostics_for_config_entry(hass, hass_client, loaded_entry)

    assert result["status"] == STATUS
    assert result["entry"]["data"] == {CONF_URL: HUB, CONF_API_TOKEN: REDACTED}
    assert TOKEN not in json.dumps(result)
    assert aioclient_mock.mock_calls[-1][3]["Authorization"] == f"Bearer {TOKEN}"


async def test_diagnostics_redacts_status_and_survives_outage(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """A token echoed by the hub is redacted; an unreachable hub is reported."""
    aioclient_mock.clear_requests()
    aioclient_mock.get(STATUS_URL, json={**STATUS, "token": TOKEN})
    result = await async_get_config_entry_diagnostics(hass, loaded_entry)
    assert result["status"] == {**STATUS, "token": REDACTED}
    assert TOKEN not in json.dumps(result)

    aioclient_mock.clear_requests()
    aioclient_mock.get(STATUS_URL, exc=ClientConnectionError("connection refused"))
    result = await async_get_config_entry_diagnostics(hass, loaded_entry)
    assert result["status"] == {"error": "connection refused"}
    assert TOKEN not in json.dumps(result)
