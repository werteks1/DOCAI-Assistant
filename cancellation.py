"""Отмена одной операции без воздействия на последующие запросы."""
import asyncio
import threading


class ExtractionCancelled(RuntimeError):
    def __init__(self):
        super().__init__("Распознавание отменено")


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def check(self):
        if self._event.is_set():
            raise ExtractionCancelled()

    async def run(self, operation):
        """Отменяет сетевую coroutine, включая ожидание заголовков/токенов."""
        task = asyncio.ensure_future(operation)
        try:
            while not task.done():
                self.check()
                await asyncio.wait({task}, timeout=0.05)
            self.check()
            return await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
