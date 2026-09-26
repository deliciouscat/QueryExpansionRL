"""Batch lifecycle, accumulation and failure handling; no model/objective imports.

The user's for-loop calls its own decorated forward exactly once per yielded
batch. This iterator only commits gradients when that call has completed.
"""

import sys


class TrainingLoop:
    def __init__(
        self,
        stream,
        engine,
        *,
        epochs,
        accumulation_steps,
        max_steps,
        on_boundary=None,
        should_stop=None,
    ):
        if min(epochs, accumulation_steps, max_steps) < 1:
            raise ValueError("Training limits must be positive")
        self.stream, self.engine = stream, engine
        self.epochs, self.accumulation_steps, self.max_steps = epochs, accumulation_steps, max_steps
        self.on_boundary = on_boundary or (lambda metrics, save_requested: None)
        self.should_stop = should_stop or (lambda: False)
        self._window = None
        self._awaiting = False
        self._entered = False
        self._pending = 0
        self._save_requested = False
        self._batches = iter(stream)

    def __enter__(self):
        if self._entered or self.engine.active:
            raise RuntimeError("TrainingLoop cannot be nested")
        self._entered = True
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if not self._entered:
            raise RuntimeError("Use TrainingLoop as a context manager")
        try:
            self._acknowledge()
            if self.engine.global_step >= self.max_steps or self.should_stop():
                raise StopIteration
            while self.stream.epoch < self.epochs:
                try:
                    batch = next(self._batches)
                    break
                except StopIteration:
                    self.stream.next_epoch()
                    self._batches = iter(self.stream)
            else:
                raise StopIteration
            if self._window is None:
                self._window = self.engine.window()
                self._window.__enter__()
            self._calls_before = self.engine.calls
            self._awaiting = True
            self._save_requested |= batch.save_flag
            return batch
        except StopIteration:
            raise
        except BaseException:
            self._abort(sys.exc_info())
            raise

    def _acknowledge(self):
        if not self._awaiting:
            return
        if self.engine.failed or self.engine.calls != self._calls_before + 1:
            raise RuntimeError("Call decorated forward exactly once per yielded batch")
        self._awaiting = False
        self._pending += 1
        if self._pending == self.accumulation_steps or self.stream.at_epoch_end:
            self._commit()

    def _commit(self):
        if self._window is None:
            return
        window, self._window = self._window, None
        window.__exit__(None, None, None)
        save_requested, self._save_requested = self._save_requested, False
        self._pending = 0
        self.on_boundary(dict(self.engine.last), save_requested)

    def _abort(self, exception):
        if self._window is not None:
            window, self._window = self._window, None
            window.__exit__(*exception)
        self._awaiting = False
        self._pending = 0

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is not None:
                self._abort((exc_type, exc, traceback))
            else:
                try:
                    self._acknowledge()
                    self._commit()  # A clean break flushes only batches actually consumed.
                except BaseException:
                    self._abort(sys.exc_info())
                    raise
        finally:
            self._entered = False
        return False
