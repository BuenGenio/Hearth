"""Entry setup and unload."""

from aiohttp import ClientConnectionError
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from homeassistant.config_entries import SOURCE_REAUTH, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.hearth.api import HearthClient
from custom_components.hearth.const import DOMAIN

from .conftest import HUB, STATUS_URL

SERVICES = {"search", "get_transcript", "summarise", "latest_summary"}


async def test_services_exist_without_entry(hass: HomeAssistant) -> None:
    """The actions are registered by async_setup, not by a config entry."""
    assert await async_setup_component(hass, DOMAIN, {})
    assert set(hass.services.async_services_for_domain(DOMAIN)) == SERVICES


async def test_setup_and_unload(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A set-up entry carries the client and creates no entities."""
    assert loaded_entry.state is ConfigEntryState.LOADED
    assert isinstance(loaded_entry.runtime_data, HearthClient)
    assert loaded_entry.runtime_data.url == HUB
    assert not er.async_entries_for_config_entry(
        er.async_get(hass), loaded_entry.entry_id
    )

    assert await hass.config_entries.async_unload(loaded_entry.entry_id)
    await hass.async_block_till_done()
    assert loaded_entry.state is ConfigEntryState.NOT_LOADED
    # Still registered, so automations keep validating.
    assert set(hass.services.async_services_for_domain(DOMAIN)) == SERVICES


async def test_setup_retries_when_hub_is_down(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """An unreachable hub leaves the entry in retry, not failed."""
    aioclient_mock.get(STATUS_URL, exc=ClientConnectionError("connection refused"))
    assert not await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_setup_starts_reauth_on_401(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """A rejected token fails setup and asks the user for a new one."""
    aioclient_mock.get(STATUS_URL, status=401, json={"detail": "nope"})
    assert not await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.SETUP_ERROR
    flows = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    assert [flow["context"]["source"] for flow in flows] == [SOURCE_REAUTH]
