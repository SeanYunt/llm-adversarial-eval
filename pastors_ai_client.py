"""
Adapter for the pastors.ai hosted chatbot widget, presenting the same
.messages.create() shape the rest of the harness expects (see LocalLLMClient
in conftest.py for the established pattern), so probe() works against it
unchanged.

pastors.ai has no public API docs -- this adapter was reverse-engineered
from a live browser session against a pastors.ai-hosted church chatbot
widget on 2026-08-04. See
docs/superpowers/specs/2026-08-04-church-chatbot-eval-design.md for the
full design rationale. `channel` has no hardcoded default deliberately --
pastors.ai hosts many churches on shared infrastructure, and this module
never names which one was used for development; see .env.example for how
to point it at a real target.
"""
import json
import re
from typing import Iterable

import httpx

_SESSION_TOKEN_RE = re.compile(r'widget_session_token\s*=\s*"([^"]+)"')
_CSRF_TOKEN_RE = re.compile(r'csrf_token\s*=\s*"([^"]+)"')


class PastorsAIError(RuntimeError):
    """Raised when the pastors.ai widget page or answer stream doesn't match
    the expected shape (e.g. the vendor changed their markup)."""


def extract_widget_tokens(html: str) -> tuple[str, str]:
    """
    Pull the session bearer token and CSRF token out of a pastors.ai widget
    page's inline <script> block. Both are server-rendered directly into the
    HTML on a plain GET -- no browser automation needed to mint a session.

    Returns (session_token, csrf_token). Raises PastorsAIError if either is
    missing.
    """
    session_match = _SESSION_TOKEN_RE.search(html)
    csrf_match = _CSRF_TOKEN_RE.search(html)
    if not session_match or not csrf_match:
        raise PastorsAIError(
            "Could not find widget_session_token/csrf_token in the widget page -- "
            "pastors.ai may have changed their markup."
        )
    return session_match.group(1), csrf_match.group(1)


def parse_sse_answer(lines: Iterable[str]) -> str:
    """
    Consume a pastors.ai SSE response and return the final answer text from
    the `done` event. Ignores the token-delta stream -- observed responses
    can render citation links garbled mid-stream; the `done` event carries
    the clean, final, correctly-formatted text. Returns "" if no `done`
    event is seen.
    """
    event = None
    answer = ""
    for line in lines:
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event == "done":
            try:
                payload = json.loads(line.split(":", 1)[1].strip())
            except (json.JSONDecodeError, ValueError) as exc:
                raise PastorsAIError(
                    "Could not parse the `done` event's JSON payload -- "
                    "pastors.ai may have changed their response format."
                ) from exc
            answer = payload.get("answer", "")
    return answer


_WIDGET_URL = "https://pastors.ai/widget/{channel}"
_ANSWER_URL = "https://pastors.ai/get_channel_answers/stream"


class _PastorsContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _PastorsResponse:
    def __init__(self, text: str) -> None:
        self.content = [_PastorsContent(text)]


class _PastorsMessages:
    def __init__(self, owner: "PastorsAIClient") -> None:
        self._owner = owner

    def create(
        self,
        *,
        model: str | None = None,  # noqa: ARG002 -- ignored; pastors.ai uses its own fixed model
        max_tokens: int | None = None,  # noqa: ARG002 -- ignored; no server-side token limit control
        messages: list,
        system: str | None = None,  # noqa: ARG002 -- ignored; pastors.ai has a fixed persona
        **kwargs,  # noqa: ARG002
    ) -> _PastorsResponse:
        question = str(messages[-1]["content"])
        answer = self._owner.ask(question)
        return _PastorsResponse(answer)


class PastorsAIClient:
    """
    Talks to a pastors.ai-hosted church chatbot widget directly over HTTP --
    no browser automation needed. `channel` is the widget path segment shown
    in the site's embed snippet, e.g. "@your-church-handle/1" (required, no
    default -- see .env.example). Presents the same .messages.create() shape
    as the Anthropic SDK, so probe() in conftest.py works against it
    unchanged.
    """

    def __init__(
        self,
        channel: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.channel = channel
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self._session_token: str | None = None
        self._csrf_token: str | None = None
        self.messages = _PastorsMessages(self)

    def _ensure_session(self) -> None:
        if self._session_token and self._csrf_token:
            return
        resp = self._http.get(_WIDGET_URL.format(channel=self.channel))
        resp.raise_for_status()
        self._session_token, self._csrf_token = extract_widget_tokens(resp.text)

    def ask(self, question: str) -> str:
        """
        Send one question and return the final answer text (HTML-formatted,
        matching what the widget itself renders -- including <a href>
        citation links).
        """
        self._ensure_session()
        name = self.channel.split("/")[0]
        body = {
            "question": question,
            "name": name,
            "subIndexTest": "False",
            "token": self._csrf_token,
            "autoQuestionText": "",
            "isMOE": "0",
        }
        headers = {"Authorization": f"Bearer {self._session_token}"}
        with self._http.stream("POST", _ANSWER_URL, json=body, headers=headers) as resp:
            resp.raise_for_status()
            return parse_sse_answer(resp.iter_lines())
