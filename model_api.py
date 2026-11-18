"""Model wire protocols and provider compatibility, independent of login methods."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from config import Settings

MODEL_APIS = ("auto", "openai-completions", "openai-responses", "anthropic-messages")


def parse_model_api(value: str | None) -> str:
    api = (value or "auto").strip().lower() or "auto"
    if api not in MODEL_APIS:
        raise ValueError(f"无效的模型协议 {api!r}；可选值：{' / '.join(MODEL_APIS)}")
    return api


def resolve_model_api(api: str, base_url: str, provider: str = "api") -> str:
    api = parse_model_api(api)
    if provider == "chatgpt":
        return "openai-responses"
    if api != "auto":
        return api
    url = urlsplit(base_url)
    if url.hostname == "api.anthropic.com" or "/anthropic" in url.path.split("/v1")[0]:
        return "anthropic-messages"
    if url.hostname == "api.openai.com":
        return "openai-responses"
    return "openai-completions"


@dataclass(frozen=True)
class ApiCompatibility:
    supports_developer_role: bool = True
    supports_encrypted_reasoning: bool = True
    supports_store: bool = True
    supports_reasoning_effort: bool = True
    requires_reasoning_content: bool = False
    adaptive_thinking: bool = False
    allow_empty_thinking_signature: bool = False


def parse_api_compat(value: str | dict | None) -> dict[str, bool]:
    try:
        data = json.loads(value) if isinstance(value, str) and value.strip() else value or {}
    except ValueError as exc:
        raise ValueError("DEEPSEEK_API_COMPAT 必须是 JSON 对象") from exc
    allowed = asdict(ApiCompatibility())
    if not isinstance(data, dict) or any(
        key not in allowed or not isinstance(item, bool) for key, item in data.items()
    ):
        raise ValueError("DEEPSEEK_API_COMPAT 只接受已知兼容选项，值必须是 true 或 false")
    return dict(data)


def compatibility(settings: "Settings") -> ApiCompatibility:
    host = urlsplit(settings.base_url).hostname
    deepseek = host == "api.deepseek.com"
    native_anthropic = host == "api.anthropic.com"
    adaptive = native_anthropic and any(
        name in settings.model for name in ("opus-4-6", "opus-4-7", "sonnet-4-6")
    )
    defaults = ApiCompatibility(
        supports_developer_role=not deepseek,
        supports_encrypted_reasoning=not deepseek,
        supports_store=not deepseek,
        requires_reasoning_content=deepseek,
        adaptive_thinking=adaptive,
        allow_empty_thinking_signature=deepseek,
    )
    return replace(defaults, **parse_api_compat(getattr(settings, "api_compat", {})))


def protocol_origin(settings: "Settings") -> str:
    """Opaque replay state is only valid for the exact endpoint, protocol and model."""
    api = resolve_model_api(
        getattr(settings, "api", "auto"), settings.base_url, getattr(settings, "provider", "api")
    )
    identity = (api, settings.base_url.rstrip("/"), settings.model)
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def message_signature(message: dict) -> str:
    values = {key: value for key, value in message.items() if key != "_harness"}
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def replay_state(message: dict, settings: "Settings") -> dict:
    metadata = message.get("_harness") or {}
    if metadata.get("protocol_origin") == protocol_origin(settings) and metadata.get(
        "protocol_message"
    ) == message_signature(message):
        return metadata
    return {}


def model_endpoint(base_url: str, api: str, operation: str | None = None) -> str:
    url = urlsplit(base_url)
    path = url.path.rstrip("/")
    if api == "anthropic-messages":
        if url.hostname == "api.deepseek.com" and path in {"", "/v1"}:
            path = "/anthropic"
        if not path.endswith("/v1"):
            path += "/v1"
        path += "/" + (operation or "messages")
    else:
        path += "/" + (
            operation or ("responses" if api == "openai-responses" else "chat/completions")
        )
    return urlunsplit((url.scheme, url.netloc, path, "", ""))
