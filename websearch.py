"""Cline's native web search endpoint, shared by Cline and ClinePass."""

import http.client
import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from config import Settings
from http_transport import build_no_redirect_opener
from tool_runtime import ExecutionContext, ToolResult


class WebSearchClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.opener = build_no_redirect_opener()

    @staticmethod
    def _domains(
        allowed_domains: list[str] | None, blocked_domains: list[str] | None
    ) -> tuple[list[str], list[str]]:
        allowed = list(
            dict.fromkeys(domain.strip() for domain in allowed_domains or [] if domain.strip())
        )
        blocked = list(
            dict.fromkeys(domain.strip() for domain in blocked_domains or [] if domain.strip())
        )
        return allowed, blocked

    def _request(self, body: dict[str, Any], api_key: str) -> urllib.request.Request:
        return urllib.request.Request(
            self.settings.websearch_base_url.rstrip("/") + "/search/websearch",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )

    def _fetch(self, request: urllib.request.Request, execution: ExecutionContext) -> Any:
        try:
            with self.opener.open(request, timeout=execution.remaining()) as response:
                payload = json.load(response)
            execution.remaining()
            return payload
        except urllib.error.HTTPError as exc:
            exc.close()
            # Gateway errors can include request credentials; do not echo bodies.
            return ToolResult.failure(
                f"http_{exc.code}",
                f"Cline web search returned HTTP {exc.code}; check the endpoint and Cline API key",
            )
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("Cline web search request timed out") from exc
            return ToolResult.failure(
                "connection_error", f"Cannot reach Cline web search ({type(exc.reason).__name__})"
            )
        except TimeoutError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            return ToolResult.failure(
                "transport_error", f"Cline web search connection failed ({type(exc).__name__})"
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ToolResult.failure(
                "invalid_response", "Cline web search did not return valid JSON"
            )

    def search(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        timeout: int = 30,
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult.failure("invalid_arguments", "query must not be blank")
        allowed, blocked = self._domains(allowed_domains, blocked_domains)
        if allowed and blocked:
            return ToolResult.failure(
                "invalid_arguments", "Use allowed_domains or blocked_domains, not both"
            )
        api_key = self.settings.resolved_websearch_api_key
        if not api_key:
            return ToolResult.failure(
                "websearch_not_configured", "Set DEEPSEEK_WEBSEARCH_API_KEY to a Cline API key"
            )
        execution = _context.child(min(timeout, self.settings.request_timeout_seconds))
        execution.remaining()
        body: dict[str, Any] = {"query": query}
        if allowed:
            body["allowed_domains"] = allowed
        if blocked:
            body["blocked_domains"] = blocked
        payload = self._fetch(self._request(body, api_key), execution)
        if isinstance(payload, ToolResult):
            return payload
        result = self._parse(payload, query)
        result.data["searched_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        for key in ("allowed_domains", "blocked_domains"):
            if key in body:
                result.data[key] = body[key]
        return result

    @staticmethod
    def _parse(payload: Any, query: str) -> ToolResult:
        if not isinstance(payload, dict):
            return ToolResult.failure("invalid_response", "Expected a Cline search response object")
        if payload.get("success") is False or payload.get("error"):
            return ToolResult.failure("provider_error", "Cline web search returned an error")
        data = payload.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return ToolResult.failure(
                "invalid_response", "Cline search response has no results array"
            )
        if data.get("error"):
            return ToolResult.failure("provider_error", "Cline web search returned an error")
        sources = []
        invalid = 0
        for item in data["results"]:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                invalid += 1
                continue
            try:
                url = urlsplit(item["url"])
                valid_url = url.scheme in {"https", "http"} and bool(url.hostname)
            except ValueError:
                valid_url = False
            if not valid_url:
                invalid += 1
                continue
            source = {"url": item["url"]}
            for key in ("title", "snippet", "description"):
                if isinstance(item.get(key), str):
                    source[key] = item[key]
            sources.append(source)
        result_data = {"provider": "cline", "query": query, "sources": sources}
        if isinstance(data.get("durationMs"), (int, float)):
            result_data["duration_ms"] = data["durationMs"]
        if invalid:
            return ToolResult.failure(
                "invalid_search_results",
                f"{invalid} search result(s) had no valid HTTP(S) URL",
                status="partial" if sources else "error",
                data=result_data,
            )
        return ToolResult(data=result_data)
