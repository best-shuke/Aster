"""Saved login choice and ChatGPT (Plus / Pro) account authorization.

The auth file holds the chosen provider and its credentials, readable only by the user.
ChatGPT access tokens rotate and each refresh token is single-use, so every client fork
shares one ``ChatGPTAccount`` per file, and a refresh re-reads the file under a lock
first: another Aster process may already have spent the refresh token.
"""

import base64
import contextlib
import fcntl
import hashlib
import http.server
import json
import os
import secrets
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CHATGPT_BASE_URL, Login
from http_transport import build_no_redirect_opener

ISSUER = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # The public Codex PKCE client.
TOKEN_URL = f"{ISSUER}/oauth/token"
SCOPE = "openid profile email offline_access"
CALLBACK_PORT = 1455
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/auth/callback"
DEVICE_URL = f"{ISSUER}/codex/device"
DEVICE_REDIRECT_URI = f"{ISSUER}/deviceauth/callback"
DEVICE_TTL_SECONDS = 15 * 60
# The backend filters its model roster by client version; older versions list fewer models.
CODEX_CLIENT_VERSION = "0.163.0"
AUTH_NAMESPACE = "https://api.openai.com/auth"
REFRESH_MARGIN_SECONDS = 300
# The auth endpoints sit behind Cloudflare, which rejects urllib's default User-Agent.
USER_AGENT = "Aster"
HTTP_TIMEOUT_SECONDS = 30


class AuthError(RuntimeError):
    """A login or token operation failed; the message is safe to show."""


class LoginExpired(AuthError):
    pass


def _now() -> float:
    return time.time()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def jwt_claims(token: str | None) -> dict[str, Any]:
    """Decode a JWT payload locally; used only for routing labels, never as proof."""
    parts = (token or "").split(".")
    if len(parts) != 3:
        return {}
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, UnicodeDecodeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _auth_claim(claims: dict[str, Any], key: str) -> str:
    namespace = claims.get(AUTH_NAMESPACE)
    value = namespace.get(key) if isinstance(namespace, dict) else None
    return value.strip() if isinstance(value, str) else ""


def account_tokens(payload: dict[str, Any], previous: dict[str, Any] | None = None) -> dict:
    """Normalize a token endpoint response, keeping fields a refresh may omit."""
    previous = previous or {}
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise AuthError("ChatGPT 授权响应缺少 access token")
    id_token = payload.get("id_token") if isinstance(payload.get("id_token"), str) else None
    id_token = id_token or previous.get("id_token") or ""
    refresh = payload.get("refresh_token")
    refresh = refresh if isinstance(refresh, str) and refresh else previous.get("refresh_token")
    if not refresh:
        raise AuthError("ChatGPT 授权响应缺少 refresh token")
    access_claims, id_claims = jwt_claims(access), jwt_claims(id_token)
    account_id = (
        _auth_claim(id_claims, "chatgpt_account_id")
        or _auth_claim(access_claims, "chatgpt_account_id")
        or previous.get("account_id", "")
    )
    if not account_id:
        raise AuthError("ChatGPT 授权响应中没有账户信息；请确认使用的是 ChatGPT 账户")
    expires = access_claims.get("exp")
    if not isinstance(expires, (int, float)):
        lifetime = payload.get("expires_in")
        expires = _now() + (lifetime if isinstance(lifetime, (int, float)) else 3600)
    email = id_claims.get("email") if isinstance(id_claims.get("email"), str) else None
    return {
        "access_token": access,
        "refresh_token": refresh,
        "id_token": id_token,
        "account_id": account_id,
        "email": email or previous.get("email", ""),
        "plan": _auth_claim(access_claims, "chatgpt_plan_type") or previous.get("plan", ""),
        "expires_at": int(expires),
        "last_refresh": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def account_label(tokens: dict[str, Any]) -> str:
    plan = (tokens.get("plan") or "").strip()
    return " · ".join(part for part in (tokens.get("email"), plan.title()) if part)


def _open(request: urllib.request.Request, timeout: float = HTTP_TIMEOUT_SECONDS):
    return build_no_redirect_opener().open(request, timeout=timeout)


def _post(url: str, fields: dict[str, str], *, form: bool = True) -> dict[str, Any]:
    if form:
        data = urllib.parse.urlencode(fields).encode()
        content_type = "application/x-www-form-urlencoded"
    else:
        data = json.dumps(fields).encode()
        content_type = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": content_type, "User-Agent": USER_AGENT},
        method="POST",
    )
    with _open(request) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise AuthError("ChatGPT 授权服务返回了无法识别的响应")
    return payload


def _oauth_error(exc: urllib.error.HTTPError) -> str:
    """The OAuth error code only; descriptions may echo request material."""
    try:
        payload = json.loads(exc.read(4096) or b"{}")
    except ValueError:
        payload = {}
    finally:
        exc.close()
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        error = error.get("code") or error.get("type")
    return f"HTTP {exc.code}" + (f" {error}" if isinstance(error, str) else "")


def exchange_code(code: str, verifier: str, redirect_uri: str) -> dict[str, Any]:
    try:
        payload = _post(
            TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": CLIENT_ID,
                "code_verifier": verifier,
            },
        )
    except urllib.error.HTTPError as exc:
        raise AuthError(f"ChatGPT 授权码兑换失败（{_oauth_error(exc)}）；请重新登录") from exc
    except (OSError, ValueError) as exc:
        raise AuthError(f"无法连接 ChatGPT 授权服务：{exc}") from exc
    return account_tokens(payload)


def refresh_tokens(tokens: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = _post(
            TOKEN_URL,
            {
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": tokens.get("refresh_token") or "",
            },
        )
    except urllib.error.HTTPError as exc:
        reason = _oauth_error(exc)
        if exc.code in {400, 401, 403}:
            raise LoginExpired(f"ChatGPT 登录已失效（{reason}），请运行 /login 重新登录") from exc
        raise AuthError(f"刷新 ChatGPT 登录失败（{reason}）") from exc
    except (OSError, ValueError) as exc:
        raise AuthError(f"无法连接 ChatGPT 授权服务：{exc}") from exc
    return account_tokens(payload, tokens)


@dataclass(frozen=True)
class BrowserLogin:
    url: str
    state: str
    verifier: str
    redirect_uri: str = REDIRECT_URI


def browser_login() -> BrowserLogin:
    verifier = _b64url(secrets.token_bytes(64))
    state = _b64url(secrets.token_bytes(32))
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()),
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": "aster",
        }
    )
    return BrowserLogin(f"{ISSUER}/oauth/authorize?{query}", state, verifier)


def parse_callback(text: str, state: str) -> str:
    """Accept a pasted redirect URL, its query string, or a bare authorization code."""
    text = text.strip()
    if not text:
        raise ValueError("没有输入内容")
    query = urllib.parse.urlsplit(text).query if "://" in text else text.lstrip("?")
    fields = urllib.parse.parse_qs(query) if "=" in query else {}
    if not fields:
        if any(character in text for character in " /?&#"):
            raise ValueError("无法从输入中找到授权码，请粘贴浏览器地址栏中的完整网址")
        return text
    if fields.get("error"):
        raise ValueError(f"授权被拒绝：{fields['error'][0]}")
    code = (fields.get("code") or [""])[0]
    if not code:
        raise ValueError("网址中没有授权码（code=...），请确认已完成登录")
    if (fields.get("state") or [state])[0] != state:
        raise ValueError("网址来自另一次登录（state 不匹配），请使用本次打开的授权链接")
    return code


_SUCCESS_PAGE = (
    "<!doctype html><meta charset=utf-8><title>Aster</title>"
    "<body style='font-family:system-ui;padding:3em'>"
    "<h2>登录成功</h2><p>可以关闭此页面，回到终端继续。</p></body>"
).encode()


class CallbackServer:
    """Receive the browser redirect on localhost:1455 when this machine runs the browser."""

    def __init__(self, state: str) -> None:
        self.state = state
        self.code: str | None = None
        self.error: str | None = None
        self.received = threading.Event()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urllib.parse.urlsplit(self.path)
                if url.path != "/auth/callback":
                    self.send_error(404)
                    return
                try:
                    owner.code = parse_callback(url.query, owner.state)
                except ValueError as exc:
                    owner.error = str(exc)
                    self.send_error(400, "Login failed")
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(_SUCCESS_PAGE)))
                    self.end_headers()
                    self.wfile.write(_SUCCESS_PAGE)
                owner.received.set()

            def log_message(self, *args) -> None:
                pass

        try:
            self.server: http.server.HTTPServer | None = http.server.HTTPServer(
                ("127.0.0.1", CALLBACK_PORT), Handler
            )
        except OSError:
            self.server = None
            return
        threading.Thread(
            target=self.server.serve_forever, name="login-callback", daemon=True
        ).start()

    @property
    def listening(self) -> bool:
        return self.server is not None

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


@dataclass(frozen=True)
class DeviceCode:
    device_auth_id: str
    user_code: str
    interval: float
    expires_at: float
    url: str = DEVICE_URL


def request_device_code() -> DeviceCode:
    try:
        payload = _post(
            f"{ISSUER}/api/accounts/deviceauth/usercode", {"client_id": CLIENT_ID}, form=False
        )
    except urllib.error.HTTPError as exc:
        raise AuthError(f"无法获取设备码（{_oauth_error(exc)}）；可以改用浏览器登录") from exc
    except (OSError, ValueError) as exc:
        raise AuthError(f"无法连接 ChatGPT 授权服务：{exc}") from exc
    device_id = payload.get("device_auth_id")
    user_code = payload.get("user_code") or payload.get("usercode")
    if not isinstance(device_id, str) or not isinstance(user_code, str):
        raise AuthError("设备码响应缺少必要字段")
    try:
        interval = float(payload.get("interval") or 5)
    except (TypeError, ValueError):
        interval = 5.0
    return DeviceCode(
        device_id, user_code, min(60.0, max(1.0, interval)), _now() + DEVICE_TTL_SECONDS
    )


def poll_device_code(device: DeviceCode) -> dict[str, Any] | None:
    """One poll: tokens once the user approved, None while still pending."""
    try:
        payload = _post(
            f"{ISSUER}/api/accounts/deviceauth/token",
            {"device_auth_id": device.device_auth_id, "user_code": device.user_code},
            form=False,
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404}:  # Pending is signalled by status, not by a body.
            exc.close()
            return None
        raise AuthError(f"设备码授权失败（{_oauth_error(exc)}）") from exc
    except (OSError, ValueError) as exc:
        raise AuthError(f"无法连接 ChatGPT 授权服务：{exc}") from exc
    code, verifier = payload.get("authorization_code"), payload.get("code_verifier")
    if not isinstance(code, str) or not isinstance(verifier, str):
        raise AuthError("设备码授权响应缺少必要字段")
    return exchange_code(code, verifier, DEVICE_REDIRECT_URI)


def list_models(tokens: dict[str, Any]) -> list[dict[str, Any]]:
    """The models this account may use, in the backend's own recommended order."""
    request = urllib.request.Request(
        f"{CHATGPT_BASE_URL}/models?client_version={CODEX_CLIENT_VERSION}",
        headers={
            "Authorization": f"Bearer {tokens['access_token']}",
            "ChatGPT-Account-Id": tokens["account_id"],
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    for attempt in range(3):
        try:
            with _open(request, timeout=20) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            exc.close()
            raise AuthError(f"无法获取模型列表（HTTP {exc.code}）") from exc
        except (OSError, ValueError) as exc:
            # The roster is large (~0.5 MB); a stalled read is usually transient.
            if attempt == 2:
                raise AuthError(f"无法获取模型列表：{exc}") from exc
    rows = payload.get("models") if isinstance(payload, dict) else None
    models = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("slug"), str):
            continue
        if row.get("visibility", "list") != "list" or row.get("supported_in_api") is False:
            continue
        window = row.get("context_window")
        # Codex keeps 5% of the window free when the roster does not say otherwise.
        percent = row.get("effective_context_window_percent") or 95
        levels = [
            level.get("effort")
            for level in row.get("supported_reasoning_levels") or []
            if isinstance(level, dict) and isinstance(level.get("effort"), str)
        ]
        models.append(
            {
                "slug": row["slug"],
                "name": row.get("display_name") or row["slug"],
                "description": row.get("description") or "",
                "context_window": int(window * percent / 100)
                if isinstance(window, int) and isinstance(percent, (int, float))
                else None,
                "reasoning_levels": levels,
                "priority": row.get("priority") if isinstance(row.get("priority"), int) else 999,
            }
        )
    return sorted(models, key=lambda model: model["priority"])


def check_api(base_url: str, api_key: str) -> tuple[int | None, list[str]]:
    """Probe an OpenAI-compatible ``/models``: (HTTP status or None, model ids)."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with _open(request, timeout=15) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        exc.close()
        return exc.code, []
    except (OSError, ValueError):
        return None, []
    rows = payload.get("data") if isinstance(payload, dict) else payload
    ids = [
        row["id"] for row in rows or [] if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    return 200, ids


class AuthStore:
    """The auth file: one JSON object, written atomically with owner-only permissions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise AuthError(f"登录信息文件无法读取：{self.path}（{exc}）") from exc
        return data if isinstance(data, dict) else {}

    def write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle, temporary = tempfile.mkstemp(prefix=".auth-", dir=self.path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise

    @contextlib.contextmanager
    def locked(self):
        """Serialize read-modify-write across Aster processes."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(self.path.with_name(self.path.name + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def update(self, **sections: Any) -> dict[str, Any]:
        """Merge top-level keys; a value of None removes that key."""
        with self.locked():
            data = self.read()
            for key, value in sections.items():
                if value is None:
                    data.pop(key, None)
                else:
                    data[key] = value
            data["version"] = 1
            self.write(data)
            return data

    def login(self) -> Login | None:
        data = self.read()
        provider = data.get("provider")
        if provider == "env":
            return Login("env")
        if provider == "api":
            api = data.get("api") or {}
            if api.get("base_url") and api.get("api_key") and api.get("model"):
                return Login("api", api["model"], api["base_url"], api["api_key"])
        if provider == "chatgpt":
            account = data.get("chatgpt") or {}
            if account.get("refresh_token") and account.get("model"):
                info = model_info(account, account["model"])
                return Login(
                    "chatgpt",
                    account["model"],
                    context_window=info.get("context_window"),
                    account=account_label(account),
                )
        return None


def model_info(account: dict[str, Any], slug: str) -> dict[str, Any]:
    for model in account.get("models") or []:
        if isinstance(model, dict) and model.get("slug") == slug:
            return model
    return {}


class ChatGPTAccount:
    """Fresh access tokens for one auth file, shared by every client in this process."""

    def __init__(self, store: AuthStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._tokens: dict[str, Any] | None = None
        self._levels: dict[str, list[str]] = {}
        # Latest plan usage reported by the backend's response headers.
        self.quota: dict[str, Any] | None = None

    def _saved(self) -> dict[str, Any]:
        account = self.store.read().get("chatgpt") or {}
        if not account.get("refresh_token") or not account.get("access_token"):
            raise LoginExpired("没有 ChatGPT 登录信息，请运行 /login 登录")
        return account

    def reset(self) -> None:
        with self._lock:
            self._tokens = None
            self._levels.clear()
            self.quota = None

    @staticmethod
    def _fresh(tokens: dict[str, Any]) -> bool:
        return tokens.get("expires_at", 0) - _now() > REFRESH_MARGIN_SECONDS

    def credentials(self) -> tuple[str, str]:
        """(access token, account id), refreshed first when it is about to expire."""
        with self._lock:
            tokens = self._tokens or self._saved()
            if not self._fresh(tokens):
                tokens = self._refresh(tokens["access_token"])
            self._tokens = tokens
            return tokens["access_token"], tokens["account_id"]

    def refresh(self, rejected_token: str) -> None:
        """Replace a token the backend rejected, unless another client already did."""
        with self._lock:
            self._tokens = self._refresh(rejected_token)

    def _refresh(self, rejected_token: str) -> dict[str, Any]:
        with self.store.locked():
            data = self.store.read()
            saved = data.get("chatgpt") or {}
            if not saved.get("refresh_token"):
                raise LoginExpired("没有 ChatGPT 登录信息，请运行 /login 登录")
            if saved.get("access_token") != rejected_token and self._fresh(saved):
                return saved  # Another client or process refreshed already.
            tokens = refresh_tokens(saved)
            data["chatgpt"] = {**saved, **tokens}
            self.store.write(data)
            return data["chatgpt"]

    def reasoning_levels(self, model: str) -> list[str]:
        if model not in self._levels:
            levels: list[str] = []
            with contextlib.suppress(AuthError):
                account = self.store.read().get("chatgpt") or {}
                levels = list(model_info(account, model).get("reasoning_levels") or [])
            self._levels[model] = levels
        return self._levels[model]


_accounts: dict[Path, ChatGPTAccount] = {}
_accounts_lock = threading.Lock()


def chatgpt_account(path: Path) -> ChatGPTAccount:
    with _accounts_lock:
        account = _accounts.get(path)
        if account is None:
            account = _accounts[path] = ChatGPTAccount(AuthStore(path))
        return account


def forget_account(path: Path) -> None:
    """Drop cached tokens after a login change, so the next request reads the file."""
    with _accounts_lock:
        account = _accounts.get(path)
    if account is not None:
        account.reset()
