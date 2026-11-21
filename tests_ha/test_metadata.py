"""The descriptive files agree with each other and with what Home Assistant loads."""

import json
from pathlib import Path
import re

from awesomeversion import AwesomeVersion
import probatio
import pytest
import yaml

from homeassistant.core import HomeAssistant
from homeassistant.helpers import service
from homeassistant.helpers.icon import async_get_icons
from homeassistant.helpers.translation import async_get_translations
from homeassistant.setup import async_setup_component

from custom_components.hearth import services as hearth_services
from custom_components.hearth.const import DOMAIN

ROOT = Path(__file__).parent.parent
INTEGRATION = ROOT / "custom_components" / DOMAIN
SCHEMAS = {
    "search": hearth_services.SEARCH_SCHEMA,
    "get_transcript": hearth_services.GET_TRANSCRIPT_SCHEMA,
    "summarise": hearth_services.SUMMARISE_SCHEMA,
    "latest_summary": hearth_services.LATEST_SUMMARY_SCHEMA,
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest() -> None:
    """What hassfest and HACS ask of the manifest."""
    manifest = _json(INTEGRATION / "manifest.json")
    keys = list(manifest)
    # hassfest: "domain, name, then alphabetical order".
    assert keys == ["domain", "name", *sorted(k for k in keys if k not in ("domain", "name"))]
    assert manifest["domain"] == DOMAIN == INTEGRATION.name
    assert manifest["config_flow"] is True
    assert manifest["requirements"] == []
    assert AwesomeVersion(manifest["version"]).valid
    # HACS: these six must be there.
    assert {"domain", "documentation", "issue_tracker", "codeowners", "name", "version"} <= set(keys)


def test_hacs_json() -> None:
    """hacs.json names the integration and a minimum Home Assistant version."""
    hacs = _json(ROOT / "hacs.json")
    assert hacs["name"] == "Hearth"
    assert "content_in_root" not in hacs
    # 2026.9 is the first release that ships probatio, which the integration imports.
    assert AwesomeVersion(hacs["homeassistant"]) >= AwesomeVersion("2026.9.0")


def test_brand_icon() -> None:
    """HACS wants brand/icon.png; it is a PNG of 256 x 256 (and 512 x 512 at 2x)."""
    for name, size in (("icon.png", 256), ("icon@2x.png", 512)):
        data = (INTEGRATION / "brand" / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert (int.from_bytes(data[16:20]), int.from_bytes(data[20:24])) == (size, size)


def test_translations_are_identical() -> None:
    """strings.json and translations/en.json carry the same content."""
    assert (INTEGRATION / "strings.json").read_bytes() == (
        INTEGRATION / "translations" / "en.json"
    ).read_bytes()


def test_exception_keys_are_translated() -> None:
    """Every translation_key raised in the code has a message, and vice versa."""
    used: set[str] = set()
    for source in INTEGRATION.glob("*.py"):
        used |= set(re.findall(r'translation_key="(\w+)"', source.read_text()))
    assert used == set(_json(INTEGRATION / "strings.json")["exceptions"])


def test_config_flow_strings() -> None:
    """Every error and abort the flow can produce has a string."""
    config = _json(INTEGRATION / "strings.json")["config"]
    assert set(config["error"]) == {"cannot_connect", "invalid_auth", "invalid_url", "unknown"}
    assert set(config["abort"]) == {"already_configured", "reauth_successful"}
    assert set(config["step"]) == {"user", "reauth_confirm"}
    for step in config["step"].values():
        assert set(step["data_description"]) <= set(step["data"])


@pytest.mark.parametrize("name", sorted(SCHEMAS))
async def test_service_is_fully_described(hass: HomeAssistant, name: str) -> None:
    """Schema, services.yaml, translations and icons list the same fields."""
    assert await async_setup_component(hass, DOMAIN, {})
    fields = {str(key) for key in SCHEMAS[name].schema}

    raw = yaml.safe_load((INTEGRATION / "services.yaml").read_text())
    assert set(raw) == set(SCHEMAS)
    assert set(raw[name]["fields"]) == fields
    for key in SCHEMAS[name].schema:
        described = raw[name]["fields"][str(key)]
        assert described.get("required", False) is isinstance(key, probatio.Required)
        if isinstance(key, probatio.Optional) and callable(key.default):
            assert described["default"] == key.default()
        else:
            assert "default" not in described

    # As Home Assistant itself loads them (this validates services.yaml too).
    described = (await service.async_get_all_descriptions(hass))[DOMAIN][name]
    assert set(described["fields"]) == fields
    assert "response" in described

    strings = await async_get_translations(hass, "en", "services", [DOMAIN])
    prefix = f"component.{DOMAIN}.services.{name}"
    assert strings[f"{prefix}.name"]
    assert strings[f"{prefix}.description"]
    for field in fields:
        assert strings[f"{prefix}.fields.{field}.name"]
        assert strings[f"{prefix}.fields.{field}.description"]
    translated = {
        key.removeprefix(f"{prefix}.fields.").split(".")[0]
        for key in strings
        if key.startswith(f"{prefix}.fields.")
    }
    assert translated == fields

    icons = await async_get_icons(hass, "services", [DOMAIN])
    assert icons[DOMAIN][name]["service"].startswith("mdi:")
