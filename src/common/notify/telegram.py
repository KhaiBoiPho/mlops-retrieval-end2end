import requests
from typing import Optional

from src.common.logging_config import get_logger

logger = get_logger(__name__)

def send_telegram_message(
    token: str,
    chat_id: str,
    text: str,
    parse_mode: str = "Markdown",
    disable_web_page_preview: bool = True,
    timeout: int = 10,
) -> Optional[dict]:
    """
    Send a message via Telegram Bot API.

    Args:
        token: Telegram bot token
        chat_id: Chat ID or group ID
        text: Message content
        parse_mode: Markdown or HTML
        disable_web_page_preview: Disable link previews
        timeout: HTTP request timeout (seconds)

    Returns:
        Telegram API response JSON if successful, None otherwise.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_page_preview,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        logger.exception("Failed to send Telegram message")
        return None
