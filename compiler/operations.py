"""Владение отменяемыми одиночными запросами; регистрация под extraction lock."""
import threading
from contextlib import contextmanager

from cancellation import CancellationToken


class OperationStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = {}

    @contextmanager
    def track(self, operation_id, owner_id):
        token = CancellationToken()
        with self._lock:
            self._active[operation_id] = (owner_id, token)
        try:
            yield token
        finally:
            with self._lock:
                self._active.pop(operation_id, None)

    def cancel(self, operation_id, owner_id):
        with self._lock:
            operation = self._active.get(operation_id)
            if operation is None or operation[0] != owner_id:
                return False
            operation[1].cancel()
            return True
