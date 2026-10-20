"""Choose how Aster reaches a model: a ChatGPT account or a third-party API."""

import os
import sys
import time
import webbrowser
from contextlib import suppress
from urllib.parse import urlsplit

from auth import (
    AuthError,
    AuthStore,
    CallbackServer,
    LoginExpired,
    account_label,
    browser_login,
    chatgpt_account,
    check_api,
    exchange_code,
    forget_account,
    list_models,
    parse_callback,
    poll_device_code,
    request_device_code,
)
from chatgpt_client import ChatGPTClient
from config import Login, Settings, load_env
from deepseek_client import DeepSeekClient
from ui import Console

DEFAULT_API_URL = "https://api.deepseek.com"
DEFAULT_API_MODEL = "deepseek-chat"


def describe(settings: Settings) -> str:
    """One line for banners and status: which service answers and as whom."""
    if settings.provider == "chatgpt":
        return "ChatGPT" + (f" · {settings.account}" if settings.account else "")
    base_url = str(settings.base_url)
    return f"API · {urlsplit(base_url).hostname or base_url}"


def model_client(settings: Settings) -> DeepSeekClient:
    return (ChatGPTClient if settings.provider == "chatgpt" else DeepSeekClient)(settings)


def _saved(store: AuthStore, section: str) -> dict:
    try:
        return store.read().get(section) or {}
    except AuthError:
        return {}


def _environment_api() -> tuple[str, str] | None:
    """(model, host) of the API configured through DEEPSEEK_* variables, if any."""
    load_env()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return None
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_API_URL)
    return (
        os.environ.get("DEEPSEEK_MODEL", DEFAULT_API_MODEL),
        urlsplit(base_url).hostname or base_url,
    )


def _remote_session() -> bool:
    """A browser opened here would not reach the user (SSH, no display)."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True
    return sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def run_login(console: Console, store: AuthStore, *, first_run: bool = False) -> Login | None:
    """Ask how to sign in and save the choice; None when the user cancels."""
    try:
        data = store.read()
    except AuthError as exc:
        console.warning(f"{exc}；将重新登录并覆盖该文件。")
        data = {}
    chatgpt = data.get("chatgpt") or {}
    environment = _environment_api()
    options = [
        {
            "label": "ChatGPT 账户（Plus / Pro）",
            "description": "使用 ChatGPT 订阅额度；在浏览器或任意设备上授权登录。"
            + (f" 已登录：{account_label(chatgpt)}" if chatgpt.get("refresh_token") else ""),
        },
        {
            "label": "第三方 API（OpenAI 兼容）",
            "description": "填写服务地址、API Key 和模型，如 DeepSeek、OpenRouter 或自建网关。",
        },
    ]
    if environment is not None:
        options.append(
            {
                "label": "沿用环境变量配置",
                "description": f"{environment[0]} · {environment[1]}"
                "（DEEPSEEK_API_KEY 等，来自 .env 或终端环境）",
            }
        )
    provider = data.get("provider") or ("env" if environment is not None else None)
    current = {"chatgpt": 0, "api": 1, "env": 2}.get(provider, 0)
    while True:
        choice = console.select_option(
            "首次启动请选择模型来源；之后可以随时用 /login 切换。"
            if first_run
            else "切换后从下一次提问开始生效，当前对话会保留。",
            options,
            default_index=current if current < len(options) else 0,
            title="欢迎使用 Aster · 选择登录方式" if first_run else "选择登录方式",
        )
        if choice is None:
            return None
        if choice == 2:
            store.update(provider="env")
            console.success(f"将使用环境变量中的配置：{environment[0]} · {environment[1]}")
            return Login("env")
        login = login_chatgpt(console, store) if choice == 0 else login_api(console, store)
        if login is not None:
            return login
        # A failed or cancelled step returns here, so another way can be chosen.
        current = choice


def login_chatgpt(console: Console, store: AuthStore) -> Login | None:
    saved = _saved(store, "chatgpt")
    tokens = None
    if saved.get("refresh_token"):
        keep = console.select_option(
            f"已登录 {account_label(saved) or 'ChatGPT 账户'}。",
            [
                {"label": "继续使用当前账户", "description": "只重新选择模型。"},
                {"label": "重新登录", "description": "换一个账户，或重新授权当前账户。"},
            ],
            title="ChatGPT 账户",
        )
        if keep is None:
            return None
        if keep == 0:
            try:
                chatgpt_account(store.path).credentials()  # Refreshes an expired token.
                tokens = store.read().get("chatgpt") or saved
            except LoginExpired as exc:
                console.warning(f"{exc}；请重新授权。")
    if tokens is None:
        try:
            tokens = _authorize(console)
        except AuthError as exc:
            console.error(exc)
            return None
        if tokens is None:
            console.info("已取消登录。")
            return None
    console.info("正在获取可用模型…")
    try:
        models = list_models(tokens)
    except AuthError as exc:
        console.warning(f"{exc}；可以手动输入模型名称。")
        models = saved.get("models") or []
    model = _choose_model(console, models, saved.get("model"))
    if model is None:
        console.info("已取消登录。")
        return None
    store.update(provider="chatgpt", chatgpt={**tokens, "model": model, "models": models})
    forget_account(store.path)
    login = store.login()
    console.success(f"已登录 ChatGPT：{account_label(tokens)} · 模型 {model}")
    return login


def _authorize(console: Console) -> dict | None:
    remote = _remote_session()
    choice = console.select_option(
        "选择授权方式。两种方式都只在 OpenAI 页面输入密码，Aster 只保存授权令牌。",
        [
            {
                "label": "浏览器登录",
                "description": "打开 ChatGPT 授权页登录。浏览器不在这台机器上时，"
                "登录后把地址栏中的网址粘贴回终端。",
            },
            {
                "label": "设备码登录",
                "description": "在手机或任意电脑打开 auth.openai.com/codex/device 并输入一次性代码。"
                "适合 SSH 和无图形界面的环境。",
            },
        ],
        default_index=1 if remote else 0,
        title="ChatGPT 登录",
    )
    if choice is None:
        return None
    return _device_login(console) if choice == 1 else _browser_login(console, remote)


def _browser_login(console: Console, remote: bool) -> dict | None:
    login = browser_login()
    server = CallbackServer(login.state)
    try:
        console.info("在浏览器中打开下面的链接并登录 ChatGPT：")
        print(f"\n{login.url}\n", flush=True)
        if not remote:
            with suppress(webbrowser.Error):
                webbrowser.open(login.url)
        if server.listening:
            console.info(
                "登录完成后会自动继续。若浏览器在另一台设备上，登录后跳转的页面会打不开，"
                "把该页面地址栏中的完整网址粘贴到下面。"
            )
        else:
            console.warning(
                f"本机端口 {urlsplit(login.redirect_uri).port} 被占用，无法自动接收登录结果；"
                "登录后请把浏览器地址栏中的完整网址粘贴到下面。"
            )
        while True:
            pasted = console.read_value("  回调网址 › ", until=server.received)
            if server.code:
                code = server.code
                break
            if server.error:
                console.warning(server.error)
                server.error = None
                server.received.clear()
                continue
            if pasted is None:
                return None
            try:
                code = parse_callback(pasted, login.state)
                break
            except ValueError as exc:
                console.warning(str(exc))
    finally:
        server.close()
    console.info("正在完成授权…")
    return exchange_code(code, login.verifier, login.redirect_uri)


def _device_login(console: Console) -> dict | None:
    device = request_device_code()
    console.info(f"1. 在任意设备的浏览器打开：{device.url}")
    console.info(f"2. 登录 ChatGPT 后输入代码：{device.user_code}")
    console.info("正在等待授权…（Ctrl-C 取消；15 分钟内有效）")
    try:
        while time.time() < device.expires_at:
            tokens = poll_device_code(device)
            if tokens is not None:
                return tokens
            time.sleep(device.interval)
    except KeyboardInterrupt:
        return None
    raise AuthError("设备码已过期，请重新登录")


def _choose_model(console: Console, models: list[dict], current: str | None) -> str | None:
    if not models:
        value = console.read_value(f"  模型名称 [{current or 'gpt-5.5'}] › ")
        if value is None:
            return None
        return value.strip() or current or "gpt-5.5"
    slugs = [model["slug"] for model in models]
    options = []
    for model in models:
        window = model.get("context_window")
        details = [model.get("description") or model.get("name") or ""]
        if window:
            details.append(f"上下文 {window // 1000}K")
        options.append(
            {
                "label": model["slug"] + ("（当前）" if model["slug"] == current else ""),
                "description": " · ".join(part for part in details if part),
            }
        )
    choice = console.select_option(
        "按推荐顺序列出此账户可用的模型；之后可用 /login 重新选择。",
        options,
        default_index=slugs.index(current) if current in slugs else 0,
        title="选择模型",
    )
    return None if choice is None else slugs[choice]


def _ask(console: Console, label: str, default: str, *, secret: bool = False) -> str | None:
    hint = f" [{default}]" if default and not secret else ""
    value = console.read_value(f"  {label}{hint} › ", secret=secret)
    if value is None:
        return None
    return value.strip() or default


def login_api(console: Console, store: AuthStore) -> Login | None:
    load_env()
    saved = _saved(store, "api")
    environment_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_API_URL)
    console.info("填写 OpenAI 兼容接口（/chat/completions）。直接回车使用方括号中的值。")
    while True:
        base_url = _ask(console, "服务地址", saved.get("base_url") or environment_url)
        if base_url is None:
            return None
        base_url = base_url.rstrip("/")
        if urlsplit(base_url).scheme in {"http", "https"} and urlsplit(base_url).hostname:
            break
        console.warning("服务地址需要以 http:// 或 https:// 开头，例如 https://api.deepseek.com")
    # Reuse a known key only for the same endpoint, never across providers.
    known_key = (
        saved.get("api_key")
        if saved.get("base_url") == base_url
        else os.environ.get("DEEPSEEK_API_KEY")
        if environment_url.rstrip("/") == base_url
        else None
    )
    while True:
        prompt = "API Key（输入不显示" + ("；回车沿用已有 Key）" if known_key else "）")
        api_key = _ask(console, prompt, known_key or "", secret=True)
        if api_key is None:
            return None
        if not api_key:
            console.warning("API Key 不能为空")
            continue
        console.info("正在验证…")
        status, model_ids = check_api(base_url, api_key)
        if status in {401, 403}:
            console.warning(f"服务拒绝了这个 API Key（HTTP {status}），请重新输入。")
            known_key = None
            continue
        if status == 200:
            console.success(f"连接成功，服务提供 {len(model_ids)} 个模型。")
        else:
            console.warning(
                f"无法通过 /models 验证（{f'HTTP {status}' if status else '网络错误'}），"
                "仍会保存；如果首次提问失败，可以再用 /login 修改。"
            )
        break
    default_model = saved.get("model") or os.environ.get("DEEPSEEK_MODEL") or ""
    if 1 < len(model_ids) <= 30:
        options = [{"label": model_id} for model_id in model_ids]
        choice = console.select_option(
            "服务返回的模型：",
            options,
            default_index=model_ids.index(default_model) if default_model in model_ids else 0,
            title="选择模型",
        )
        if choice is None:
            return None
        model = model_ids[choice]
    else:
        model = _ask(
            console,
            "模型名称",
            default_model or (model_ids[0] if model_ids else DEFAULT_API_MODEL),
        )
        if model is None:
            return None
        if model_ids and model not in model_ids:
            console.warning(f"{model} 不在服务返回的模型列表中，请确认名称拼写。")
    store.update(provider="api", api={"base_url": base_url, "api_key": api_key, "model": model})
    console.success(f"已保存第三方 API：{model} · {urlsplit(base_url).hostname}")
    return store.login()


def logout(console: Console, store: AuthStore) -> bool:
    """Delete saved credentials; the next start asks how to sign in again."""
    if not store.path.exists():
        console.info("没有保存的登录信息。")
        return False
    store.update(provider=None, chatgpt=None, api=None)
    forget_account(store.path)
    console.success("已退出登录并删除保存的凭据；下次启动时会重新选择登录方式。")
    return True
