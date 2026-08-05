"""
Unit tests for pastors_ai_client.py's pure parsing logic (no network calls,
always runs regardless of ANTHROPIC_API_KEY -- same category as
tests/test_build_taxonomy.py).
"""
import json

import httpx
import pytest

from pastors_ai_client import PastorsAIError, PastorsAIClient, extract_widget_tokens, parse_sse_answer


class TestExtractWidgetTokens:
    def test_extracts_both_tokens_from_widget_html(self) -> None:
        html = '''
        <html><body>
        <script>
            var widget_session_token = "eyJzaWQiOiJhYmMifQ.xyz.sig123";
            var csrf_token = "ImFiYyI.xyz.sig456";
        </script>
        </body></html>
        '''
        session_token, csrf_token = extract_widget_tokens(html)
        assert session_token == "eyJzaWQiOiJhYmMifQ.xyz.sig123"
        assert csrf_token == "ImFiYyI.xyz.sig456"

    def test_raises_when_session_token_missing(self) -> None:
        html = '<script>var csrf_token = "onlyone";</script>'
        with pytest.raises(PastorsAIError):
            extract_widget_tokens(html)

    def test_raises_when_csrf_token_missing(self) -> None:
        html = '<script>var widget_session_token = "onlyone";</script>'
        with pytest.raises(PastorsAIError):
            extract_widget_tokens(html)

    def test_raises_when_both_missing(self) -> None:
        html = '<html><body>no tokens here</body></html>'
        with pytest.raises(PastorsAIError):
            extract_widget_tokens(html)


class TestParseSseAnswer:
    def test_returns_done_event_answer(self) -> None:
        lines = [
            'event: token',
            'data: {"t": "Hi"}',
            '',
            'event: done',
            'data: {"answer": "Sunday services are at 9am and 10:30am."}',
            '',
        ]
        assert parse_sse_answer(lines) == "Sunday services are at 9am and 10:30am."

    def test_ignores_token_deltas_uses_only_done_event(self) -> None:
        # Regression case: real pastors.ai responses can render citation links
        # garbled across token-delta events -- only the `done` event's answer
        # is authoritative.
        lines = [
            'event: token',
            'data: {"t": "(#"}',
            'event: token',
            'data: {"t": "!!00:00 <-DF0tSPoCmM>!!#)."}',
            'event: done',
            'data: {"answer": "Clean final text with a proper link."}',
        ]
        assert parse_sse_answer(lines) == "Clean final text with a proper link."

    def test_returns_empty_string_if_no_done_event(self) -> None:
        lines = ['event: token', 'data: {"t": "partial"}']
        assert parse_sse_answer(lines) == ""

    def test_returns_empty_string_for_empty_input(self) -> None:
        assert parse_sse_answer([]) == ""

    def test_raises_pastors_ai_error_on_malformed_done_event_json(self) -> None:
        # If pastors.ai changes their response format and the `done` event's
        # data isn't valid JSON, this should surface as PastorsAIError (per
        # its docstring covering the answer stream), not a raw
        # json.JSONDecodeError leaking an implementation detail.
        lines = [
            'event: done',
            'data: {this is not valid json',
            '',
        ]
        with pytest.raises(PastorsAIError):
            parse_sse_answer(lines)


_FAKE_WIDGET_HTML = '''
<script>
    var widget_session_token = "FAKE_SESSION_TOKEN";
    var csrf_token = "FAKE_CSRF_TOKEN";
</script>
'''

_FAKE_SSE_BODY = (
    'event: token\ndata: {"t": "Sun"}\n\n'
    'event: done\ndata: {"answer": "Sunday services are at 9am and 10:30am."}\n\n'
)


def _make_client(handler) -> PastorsAIClient:
    transport = httpx.MockTransport(handler)
    return PastorsAIClient(channel="@example-church/1", transport=transport)


class TestPastorsAIClientAsk:
    def test_ask_bootstraps_session_then_posts_question(self) -> None:
        requests_seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(request)
            if request.method == "GET":
                return httpx.Response(200, text=_FAKE_WIDGET_HTML)
            return httpx.Response(200, text=_FAKE_SSE_BODY, headers={"content-type": "text/event-stream"})

        client = _make_client(handler)
        answer = client.ask("What time is service?")

        assert answer == "Sunday services are at 9am and 10:30am."
        assert len(requests_seen) == 2
        get_req, post_req = requests_seen
        assert get_req.method == "GET"
        assert str(get_req.url) == "https://pastors.ai/widget/@example-church/1"
        assert post_req.method == "POST"
        assert post_req.headers["authorization"] == "Bearer FAKE_SESSION_TOKEN"
        body = json.loads(post_req.content)
        assert body["question"] == "What time is service?"
        assert body["token"] == "FAKE_CSRF_TOKEN"
        assert body["name"] == "@example-church"

    def test_ask_reuses_session_across_multiple_calls(self) -> None:
        get_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal get_count
            if request.method == "GET":
                get_count += 1
                return httpx.Response(200, text=_FAKE_WIDGET_HTML)
            return httpx.Response(200, text=_FAKE_SSE_BODY, headers={"content-type": "text/event-stream"})

        client = _make_client(handler)
        client.ask("question one")
        client.ask("question two")

        assert get_count == 1

    def test_ask_raises_pastors_ai_error_on_malformed_widget_page(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>no tokens here</html>")

        client = _make_client(handler)
        with pytest.raises(PastorsAIError):
            client.ask("anything")


class TestPastorsAIClientMessagesCreate:
    def test_create_extracts_last_user_message_and_returns_anthropic_shaped_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, text=_FAKE_WIDGET_HTML)
            return httpx.Response(200, text=_FAKE_SSE_BODY, headers={"content-type": "text/event-stream"})

        client = _make_client(handler)
        response = client.messages.create(
            model="ignored",
            max_tokens=100,
            messages=[{"role": "user", "content": "What time is service?"}],
        )

        assert response.content[0].text == "Sunday services are at 9am and 10:30am."

    def test_create_ignores_model_max_tokens_and_system_kwargs(self) -> None:
        # pastors.ai has a fixed persona and its own fixed model -- these
        # Anthropic-shaped kwargs exist only so probe() can call this the
        # same way it calls the real Anthropic client, and must not raise.
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, text=_FAKE_WIDGET_HTML)
            return httpx.Response(200, text=_FAKE_SSE_BODY, headers={"content-type": "text/event-stream"})

        client = _make_client(handler)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system="this should be silently ignored",
            messages=[{"role": "user", "content": "What time is service?"}],
        )

        assert response.content[0].text == "Sunday services are at 9am and 10:30am."
