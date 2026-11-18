"""Select model wire protocols separately from account authentication."""

from anthropic_client import AnthropicClient
from chatgpt_client import ChatGPTClient
from config import Settings
from deepseek_client import DeepSeekClient
from responses_client import ResponsesClient

API_CLIENTS = {
    "openai-completions": DeepSeekClient,
    "openai-responses": ResponsesClient,
    "anthropic-messages": AnthropicClient,
}


def model_client(settings: Settings) -> DeepSeekClient:
    client_type = (
        ChatGPTClient if settings.provider == "chatgpt" else API_CLIENTS[settings.resolved_api]
    )
    return client_type(settings)
