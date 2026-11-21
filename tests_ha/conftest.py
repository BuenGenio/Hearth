"""Fixtures for the Hearth integration tests."""

from collections.abc import AsyncGenerator

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from homeassistant.const import CONF_API_TOKEN, CONF_URL
from homeassistant.core import HomeAssistant

from custom_components.hearth.const import DOMAIN

HUB = "http://hearth.test:8377"
TOKEN = "s3cret-token"
STATUS_URL = f"{HUB}/api/v1/status"
STATUS = {
    "version": "0.1.0",
    "segments": 42,
    "recording": True,
    "guard": False,
    "live": True,
    "mic_connected": True,
    "mic_signal": True,
    "stt_backend": "sherpa",
    "stt_queued": 0,
    "stt_dropped": 0,
}
SEGMENT = {
    "id": "0b0e7c0e-3f1c-4f0e-9d51-2f6f6f1f2a10",
    "source": "room",
    "started_at": "2026-11-20T09:15:00.000000Z",
    "ended_at": "2026-11-20T09:15:07.000000Z",
    "text": "the dentist appointment is on Tuesday",
    "language": "en",
    "backend": "sherpa",
    "created_at": "2026-11-20T09:15:09.000000Z",
    "has_audio": False,
}


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Let Home Assistant load integrations from custom_components/."""


@pytest.fixture
def config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """A Hearth config entry, added to hass but not set up."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hearth (hearth.test)",
        data={CONF_URL: HUB, CONF_API_TOKEN: TOKEN},
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def loaded_entry(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> AsyncGenerator[MockConfigEntry]:
    """A Hearth config entry that is set up against a mocked hub."""
    aioclient_mock.get(STATUS_URL, json=STATUS)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    aioclient_mock.mock_calls.clear()
    yield config_entry
