"""Config flow: success, invalid auth, cannot connect, duplicate, reauth."""

from aiohttp import ClientConnectionError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_TOKEN, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.hearth.const import DEFAULT_URL, DOMAIN

from .conftest import HUB, STATUS, STATUS_URL, TOKEN


async def _start(hass: HomeAssistant) -> str:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {}
    return result["flow_id"]


async def test_user_flow_success(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A reachable hub and a good token create the entry; the URL is tidied."""
    aioclient_mock.get(STATUS_URL, json=STATUS)
    flow_id = await _start(hass)

    result = await hass.config_entries.flow.async_configure(
        flow_id, {CONF_URL: f" {HUB}/ ", CONF_API_TOKEN: f"{TOKEN} "}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Hearth (hearth.test)"
    assert result["data"] == {CONF_URL: HUB, CONF_API_TOKEN: TOKEN}
    # The check in the flow and the one at setup both carried the bearer token.
    assert aioclient_mock.call_count == 2
    for method, url, _, headers in aioclient_mock.mock_calls:
        assert (method, str(url)) == ("GET", STATUS_URL)
        assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert hass.config_entries.async_loaded_entries(DOMAIN)


async def test_user_flow_default_url(hass: HomeAssistant) -> None:
    """The form proposes the owner's hub address."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    (url_key,) = (key for key in result["data_schema"].schema if key == CONF_URL)
    assert url_key.default() == DEFAULT_URL


async def test_user_flow_invalid_auth(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """HTTP 401 shows invalid_auth; the user can correct the token."""
    aioclient_mock.get(
        STATUS_URL, status=401, json={"detail": "invalid or missing bearer token"}
    )
    flow_id = await _start(hass)

    result = await hass.config_entries.flow.async_configure(
        flow_id, {CONF_URL: HUB, CONF_API_TOKEN: "wrong"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}
    assert not hass.config_entries.async_entries(DOMAIN)

    aioclient_mock.clear_requests()
    aioclient_mock.get(STATUS_URL, json=STATUS)
    result = await hass.config_entries.flow.async_configure(
        flow_id, {CONF_URL: HUB, CONF_API_TOKEN: TOKEN}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY


@pytest.mark.parametrize(
    "mock_kwargs",
    [
        {"exc": ClientConnectionError("connection refused")},
        {"exc": TimeoutError()},
        {"status": 500, "text": "Internal Server Error"},
        {"status": 404, "json": {"detail": "Not Found"}},
        {"text": "<html>some other web server</html>"},
        {"json": {"hello": "not a hearth hub"}},
    ],
    ids=["refused", "timeout", "http-500", "http-404", "not-json", "not-hearth"],
)
async def test_user_flow_cannot_connect(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, mock_kwargs: dict
) -> None:
    """Anything but a Hearth status answer shows cannot_connect."""
    aioclient_mock.get(STATUS_URL, **mock_kwargs)
    flow_id = await _start(hass)

    result = await hass.config_entries.flow.async_configure(
        flow_id, {CONF_URL: HUB, CONF_API_TOKEN: TOKEN}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}
    assert not hass.config_entries.async_entries(DOMAIN)


@pytest.mark.parametrize("url", ["hearth.test:8377", "ftp://hearth.test", "http://"])
async def test_user_flow_invalid_url(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, url: str
) -> None:
    """A URL without http(s) scheme or host is refused before any request."""
    flow_id = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow_id, {CONF_URL: url, CONF_API_TOKEN: TOKEN}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_URL: "invalid_url"}
    assert aioclient_mock.call_count == 0


async def test_user_flow_duplicate(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """The same hub cannot be added twice, however the URL is written."""
    flow_id = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow_id, {CONF_URL: "HTTP://Hearth.Test:8377/", CONF_API_TOKEN: "another"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert aioclient_mock.call_count == 0
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1


async def test_reauth_flow(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Reauth rejects a bad token, then stores a good one and reloads."""
    result = await loaded_entry.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["description_placeholders"]["url"] == HUB

    aioclient_mock.clear_requests()
    aioclient_mock.get(STATUS_URL, status=401, json={"detail": "nope"})
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_TOKEN: "still-wrong"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}
    assert loaded_entry.data[CONF_API_TOKEN] == TOKEN

    aioclient_mock.clear_requests()
    aioclient_mock.get(STATUS_URL, json=STATUS)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_TOKEN: "new-token"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert loaded_entry.data == {CONF_URL: HUB, CONF_API_TOKEN: "new-token"}
    assert aioclient_mock.mock_calls[-1][3]["Authorization"] == "Bearer new-token"
