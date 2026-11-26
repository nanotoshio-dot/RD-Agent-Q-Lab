import os
from collections.abc import Mapping
from pathlib import Path

MARKET_ENV_PREFIX = "RD_MARKET_"
DEFAULT_CHINA_PROVIDER = "~/.qlib/qlib_data/cn_data"


def current_market_id(env: Mapping[str, str] | None = None) -> str | None:
    """Return the market of a launcher run, or None for a plain RD-Agent run.

    Market variables without RD_MARKET_ID are rejected: the Qlib templates would follow them
    while the rest of the run kept assuming the default China setup.
    """
    env = os.environ if env is None else env
    market_id = env.get("RD_MARKET_ID", "").strip()
    if market_id:
        return market_id
    stray = sorted(key for key in env if key.startswith(MARKET_ENV_PREFIX))
    if stray:
        message = f"Market variables are set without RD_MARKET_ID: {', '.join(stray)}"
        raise ValueError(message)
    return None


def ensure_docker_supports_market(market_id: str | None, provider_uri: str | None) -> None:
    """Refuse Docker execution for a market run it would silently turn into a China run.

    The container receives none of the RD_MARKET_* variables and mounts ~/.qlib at another
    path, so its Qlib templates always render the default China setup.
    """
    if market_id is None:
        return
    if market_id == "china" and (not provider_uri or _same_path(provider_uri, DEFAULT_CHINA_PROVIDER)):
        return
    message = (
        f"Market {market_id!r} cannot run in Docker: the container does not receive the market "
        "settings and would use the default China data. Use --model-env local or conda."
    )
    raise ValueError(message)


def _same_path(left: str, right: str) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
