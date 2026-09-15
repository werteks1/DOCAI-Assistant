"""Адаптер HTTP-потока: прерывает ожидание сети по токену операции."""
import asyncio
from contextlib import contextmanager

import httpx


class StreamingResponse:
    def __init__(self, response, loop, cancellation):
        self.response = response
        self.loop = loop
        self.cancellation = cancellation
        self.status_code = response.status_code
        self.headers = response.headers

    def _run(self, operation):
        return self.loop.run_until_complete(self.cancellation.run(operation))

    def iter_lines(self):
        iterator = self.response.aiter_lines()
        while True:
            try:
                yield self._run(anext(iterator)).encode("utf-8")
            except StopAsyncIteration:
                return

    def json(self):
        self._run(self.response.aread())
        return self.response.json()

    @property
    def text(self):
        self._run(self.response.aread())
        return self.response.text

    def close(self):
        self.loop.run_until_complete(self.response.aclose())


class Transport:
    def __init__(self, client, loop, cancellation):
        self.client = client
        self.loop = loop
        self.cancellation = cancellation
        self.responses = []

    def post(self, url, *, json, stream=True, timeout=None):
        request = self.client.build_request("POST", url, json=json, timeout=timeout)
        response = self.loop.run_until_complete(
            self.cancellation.run(self.client.send(request, stream=stream))
        )
        wrapped = StreamingResponse(response, self.loop, self.cancellation)
        self.responses.append(wrapped)
        return wrapped

    def sleep(self, seconds):
        self.loop.run_until_complete(self.cancellation.run(asyncio.sleep(seconds)))


@contextmanager
def cancellable_transport(cancellation, headers):
    loop = asyncio.new_event_loop()
    client = httpx.AsyncClient(headers=headers, follow_redirects=True)
    transport = Transport(client, loop, cancellation)
    try:
        yield transport
    finally:
        try:
            for response in transport.responses:
                response.close()
        finally:
            loop.run_until_complete(client.aclose())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
