"""The local brand icon is what Home Assistant serves for the integration."""

from pathlib import Path

from pytest_homeassistant_custom_component.typing import ClientSessionGenerator

from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from custom_components.hearth.const import DOMAIN

BRAND = Path(__file__).parent.parent / "custom_components" / DOMAIN / "brand"


async def test_brand_icon_is_served(
    hass: HomeAssistant, hass_client: ClientSessionGenerator
) -> None:
    """custom_components/hearth/brand/ is picked up (Home Assistant 2026.3+)."""
    assert await async_setup_component(hass, "brands", {})
    await hass.async_block_till_done()
    client = await hass_client()

    for name in ("icon.png", "icon@2x.png"):
        response = await client.get(
            f"/api/brands/integration/{DOMAIN}/{name}", params={"placeholder": "no"}
        )
        assert response.status == 200
        assert response.content_type == "image/png"
        assert await response.read() == (BRAND / name).read_bytes()
