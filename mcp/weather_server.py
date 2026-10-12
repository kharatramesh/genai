"""MCP server #1: current weather (Open-Meteo, free, no API key)."""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather", log_level="WARNING")

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WX_URL = "https://api.open-meteo.com/v1/forecast"


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get the current weather for a city, e.g. 'Pune' or 'London'."""
    geo = httpx.get(GEO_URL, params={"name": city, "count": 1}, timeout=15).json()
    if not geo.get("results"):
        return f"City '{city}' not found."
    place = geo["results"][0]

    wx = httpx.get(WX_URL, timeout=15, params={
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
    }).json()["current"]

    return (f"{place['name']}, {place.get('country', '')}: "
            f"{wx['temperature_2m']} C, humidity {wx['relative_humidity_2m']}%, "
            f"wind {wx['wind_speed_10m']} km/h (WMO code {wx['weather_code']})")


if __name__ == "__main__":
    mcp.run()  # stdio transport
