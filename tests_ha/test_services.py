"""Service actions: happy paths and HTTP failures, hub mocked with aioclient_mock."""

from typing import Any

from aiohttp import ClientConnectionError
import probatio
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from homeassistant.config_entries import SOURCE_REAUTH
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.setup import async_setup_component

from custom_components.hearth.const import DOMAIN

from .conftest import HUB, SEGMENT, TOKEN

SEARCH_URL = f"{HUB}/api/v1/search"
SEGMENTS_URL = f"{HUB}/api/v1/segments"
RUN_URL = f"{HUB}/api/v1/summaries/run"
LATEST_URL = f"{HUB}/api/v1/summaries/latest"

RUN_SUMMARY = {"text": "A quiet morning.", "model": "claude", "segments": 12}
STORED_SUMMARY = {
    "id": 7,
    "kind": "daily",
    "window_start": "2026-11-19T23:00:00.000000Z",
    "window_end": "2026-11-20T23:00:00.000000Z",
    "text": "A quiet day.",
    "model": "claude",
    "segment_count": 80,
    "created_at": "2026-11-20T23:00:04.000000Z",
}


async def _call(hass: HomeAssistant, name: str, data: dict[str, Any]) -> Any:
    return await hass.services.async_call(
        DOMAIN, name, data, blocking=True, return_response=True
    )


def _only_call(aioclient_mock: AiohttpClientMocker) -> tuple[str, Any, Any, Any]:
    assert aioclient_mock.call_count == 1
    method, url, body, headers = aioclient_mock.mock_calls[0]
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    return method, url, body, headers


# -- search ---------------------------------------------------------------------


@pytest.mark.usefixtures("loaded_entry")
async def test_search(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """Search passes q and the default limit, and returns the segments."""
    aioclient_mock.get(SEARCH_URL, json={"segments": [SEGMENT]})

    assert await _call(hass, "search", {"query": "dentist appointment"}) == {
        "segments": [SEGMENT]
    }
    method, url, _, _ = _only_call(aioclient_mock)
    assert (method, url.path) == ("GET", "/api/v1/search")
    assert dict(url.query) == {"q": "dentist appointment", "limit": "20"}


@pytest.mark.usefixtures("loaded_entry")
async def test_search_limit_and_entry_id(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """An explicit limit and an explicit hub are honoured."""
    aioclient_mock.get(SEARCH_URL, json={"segments": []})
    response = await _call(
        hass,
        "search",
        {"query": "x", "limit": 5, "config_entry_id": loaded_entry.entry_id},
    )
    assert response == {"segments": []}
    assert dict(_only_call(aioclient_mock)[1].query) == {"q": "x", "limit": "5"}


@pytest.mark.usefixtures("loaded_entry")
async def test_search_http_error(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A server error becomes a HomeAssistantError that quotes the hub."""
    aioclient_mock.get(SEARCH_URL, status=500, text="Internal Server Error")
    with pytest.raises(HomeAssistantError) as err:
        await _call(hass, "search", {"query": "x"})
    assert not isinstance(err.value, ServiceValidationError)
    assert err.value.translation_key == "api_error"
    assert err.value.translation_placeholders == {
        "status": "500",
        "detail": "Internal Server Error",
    }


@pytest.mark.usefixtures("loaded_entry")
async def test_search_rejected_by_hub(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """HTTP 422 becomes a ServiceValidationError with the hub's reason."""
    aioclient_mock.get(
        SEARCH_URL,
        status=422,
        json={"detail": [{"loc": ["query", "q"], "msg": "String too long", "type": "x"}]},
    )
    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, "search", {"query": "x"})
    assert err.value.translation_key == "request_rejected"
    assert err.value.translation_placeholders == {"detail": "q: String too long"}


@pytest.mark.usefixtures("loaded_entry")
@pytest.mark.parametrize(
    "data",
    [{}, {"query": ""}, {"query": "x" * 201}, {"query": "x", "limit": 0}, {"query": "x", "limit": 501}],
    ids=["no-query", "empty-query", "long-query", "limit-0", "limit-501"],
)
async def test_search_schema(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, data: dict[str, Any]
) -> None:
    """Out-of-contract input is refused before the hub is asked."""
    with pytest.raises(probatio.Invalid):
        await _call(hass, "search", data)
    assert aioclient_mock.call_count == 0


async def test_search_requires_response(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Read-only actions are response-only."""
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(DOMAIN, "search", {"query": "x"}, blocking=True)


# -- get_transcript -------------------------------------------------------------


@pytest.mark.usefixtures("loaded_entry")
async def test_get_transcript(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """Times are sent to the hub in UTC; naive ones count as local time."""
    await hass.config.async_set_time_zone("Europe/Madrid")
    aioclient_mock.get(SEGMENTS_URL, json={"segments": [SEGMENT]})

    response = await _call(
        hass,
        "get_transcript",
        {"start": "2026-11-20 10:00:00", "end": "2026-11-20T11:30:00+01:00", "limit": 50},
    )
    assert response == {"segments": [SEGMENT]}
    method, url, _, _ = _only_call(aioclient_mock)
    assert (method, url.path) == ("GET", "/api/v1/segments")
    assert dict(url.query) == {
        "start": "2026-11-20T09:00:00.000000Z",
        "end": "2026-11-20T10:30:00.000000Z",
        "limit": "50",
    }


@pytest.mark.usefixtures("loaded_entry")
@pytest.mark.freeze_time("2026-11-21 12:00:00+00:00")
async def test_get_transcript_end_defaults_to_now(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """Without end the range runs until now, with the hub's default limit."""
    aioclient_mock.get(SEGMENTS_URL, json={"segments": []})
    await _call(hass, "get_transcript", {"start": "2026-11-21T11:00:00+00:00"})
    assert dict(_only_call(aioclient_mock)[1].query) == {
        "start": "2026-11-21T11:00:00.000000Z",
        "end": "2026-11-21T12:00:00.000000Z",
        "limit": "1000",
    }


@pytest.mark.usefixtures("loaded_entry")
async def test_get_transcript_end_before_start(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A reversed range is refused without asking the hub."""
    with pytest.raises(ServiceValidationError) as err:
        await _call(
            hass,
            "get_transcript",
            {"start": "2026-11-20T11:00:00+00:00", "end": "2026-11-20T10:00:00+00:00"},
        )
    assert err.value.translation_key == "end_before_start"
    assert aioclient_mock.call_count == 0


@pytest.mark.usefixtures("loaded_entry")
async def test_get_transcript_hub_unreachable(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A connection failure becomes a HomeAssistantError naming the hub."""
    aioclient_mock.get(SEGMENTS_URL, exc=ClientConnectionError("connection refused"))
    with pytest.raises(HomeAssistantError) as err:
        await _call(hass, "get_transcript", {"start": "2026-11-20T10:00:00+00:00"})
    assert err.value.translation_key == "cannot_connect"
    assert err.value.translation_placeholders == {
        "url": HUB,
        "error": "connection refused",
    }


# -- summarise ------------------------------------------------------------------


@pytest.mark.usefixtures("loaded_entry")
async def test_summarise_defaults(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """Defaults are one hour and kind 'manual'; the summary comes back."""
    aioclient_mock.post(RUN_URL, json={"summary": RUN_SUMMARY})

    assert await _call(hass, "summarise", {}) == {"summary": RUN_SUMMARY}
    method, url, body, _ = _only_call(aioclient_mock)
    assert (method, url.path) == ("POST", "/api/v1/summaries/run")
    assert body == {"kind": "manual", "hours": 1.0}


@pytest.mark.usefixtures("loaded_entry")
async def test_summarise_without_response(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """It also works as a plain action; nothing to summarise is not an error."""
    aioclient_mock.post(RUN_URL, json={"summary": None})

    result = await hass.services.async_call(
        DOMAIN, "summarise", {"hours": 6, "kind": "evening"}, blocking=True
    )
    assert result is None
    assert _only_call(aioclient_mock)[2] == {"kind": "evening", "hours": 6.0}

    aioclient_mock.mock_calls.clear()
    assert await _call(hass, "summarise", {"hours": 0.5}) == {"summary": None}


@pytest.mark.usefixtures("loaded_entry")
@pytest.mark.parametrize(
    ("status", "detail"),
    [(503, "summaries are not enabled"), (502, "summary model failed: timeout")],
)
async def test_summarise_http_error(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, status: int, detail: str
) -> None:
    """The hub's 502/503 explanations reach the user."""
    aioclient_mock.post(RUN_URL, status=status, json={"detail": detail})
    with pytest.raises(HomeAssistantError) as err:
        await _call(hass, "summarise", {})
    assert err.value.translation_key == "api_error"
    assert err.value.translation_placeholders == {"status": str(status), "detail": detail}


@pytest.mark.usefixtures("loaded_entry")
@pytest.mark.parametrize("data", [{"hours": 0}, {"hours": 73}, {"kind": "k" * 33}])
async def test_summarise_schema(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, data: dict[str, Any]
) -> None:
    """The hub's limits on hours and kind are enforced up front."""
    with pytest.raises(probatio.Invalid):
        await _call(hass, "summarise", data)
    assert aioclient_mock.call_count == 0


# -- latest_summary -------------------------------------------------------------


@pytest.mark.usefixtures("loaded_entry")
async def test_latest_summary(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """Without kind no filter is sent; with kind it is passed on."""
    aioclient_mock.get(LATEST_URL, json={"summary": STORED_SUMMARY})

    assert await _call(hass, "latest_summary", {}) == {"summary": STORED_SUMMARY}
    method, url, _, _ = _only_call(aioclient_mock)
    assert (method, url.path, dict(url.query)) == ("GET", "/api/v1/summaries/latest", {})

    aioclient_mock.mock_calls.clear()
    await _call(hass, "latest_summary", {"kind": "daily"})
    assert dict(_only_call(aioclient_mock)[1].query) == {"kind": "daily"}


@pytest.mark.usefixtures("loaded_entry")
async def test_latest_summary_none(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """No stored summary is an empty answer, not an error."""
    aioclient_mock.get(LATEST_URL, json={"summary": None})
    assert await _call(hass, "latest_summary", {}) == {"summary": None}


async def test_latest_summary_token_rejected(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """HTTP 401 raises a clear error and starts the reauth flow."""
    aioclient_mock.get(
        LATEST_URL, status=401, json={"detail": "invalid or missing bearer token"}
    )
    with pytest.raises(HomeAssistantError) as err:
        await _call(hass, "latest_summary", {})
    assert err.value.translation_key == "auth_failed"
    assert err.value.translation_placeholders == {"title": loaded_entry.title}
    await hass.async_block_till_done()
    flows = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    assert [flow["context"]["source"] for flow in flows] == [SOURCE_REAUTH]


# -- finding the hub at call time -----------------------------------------------


async def test_call_without_configured_hub(hass: HomeAssistant) -> None:
    """With no hub set up the action exists and says what is missing."""
    assert await async_setup_component(hass, DOMAIN, {})
    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, "search", {"query": "x"})
    assert err.value.translation_key == "service_found_no_config_entry_for_domain"


async def test_call_with_unloaded_hub(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """An unloaded hub is reported, and the hub is not contacted."""
    assert await hass.config_entries.async_unload(loaded_entry.entry_id)
    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, "latest_summary", {})
    assert err.value.translation_key == "service_config_entry_not_loaded"
    assert aioclient_mock.call_count == 0


async def test_call_with_unknown_entry_id(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A config_entry_id that does not exist is a validation error."""
    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, "search", {"query": "x", "config_entry_id": "nope"})
    assert err.value.translation_key == "service_config_entry_not_found"
