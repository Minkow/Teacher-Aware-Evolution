"""Bounded, reusable rollout workers with per-evaluation cancellation."""

import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import wraps
import math
import multiprocessing as mp
import os
import signal
import threading
import time


_local = threading.local()


class EvaluationBudget:
    def __init__(self, deadline):
        self.deadline = float(deadline)
        self.cancelled = threading.Event()

    def check(self):
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise TimeoutError("evaluation deadline exceeded; rollout cancelled")

    def run(self, function, *args, **kwargs):
        previous = getattr(_local, "budget", None)
        _local.budget = self
        try:
            self.check()
            result = function(*args, **kwargs)
            self.check()
            return result
        finally:
            _local.budget = previous


def bounded_evaluation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        budget = getattr(_local, "budget", None)
        if budget is None:
            seconds = float(os.environ.get("EOH_EVALUATION_TIMEOUT_S", "1800"))
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("EOH_EVALUATION_TIMEOUT_S must be finite and positive")
            budget = EvaluationBudget(time.monotonic() + seconds)
        return budget.run(function, *args, **kwargs)
    return wrapped


def _worker(connection):
    if os.name == "posix":
        os.setsid()
    try:
        connection.send(("ready", None))
        while True:
            function, args = connection.recv()
            try:
                result = function(*args)
                connection.send(("ok", result))
            except BaseException as exc:
                connection.send(("error", (type(exc).__name__, str(exc))))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class _Slot:
    def __init__(self):
        self.process = None
        self.connection = None

    def stop(self):
        process, connection = self.process, self.connection
        self.process = self.connection = None
        if process is None:
            return
        try:
            # Each worker is a session leader. Kill its descendants as well,
            # without signalling any other slot or the parent process group.
            if os.name == "posix" and process.pid is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    # macOS may report EPERM after the session leader exits.
                    # Still kill/join the owned worker below if it is alive.
                    if process.is_alive():
                        process.kill()
            if process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(timeout=3)
            if process.is_alive():
                raise RuntimeError(f"could not reap rollout worker {process.pid}")
            process.close()
        finally:
            if connection is not None:
                connection.close()

    def receive(self, budget, batch_cancelled):
        while True:
            budget.check()
            if batch_cancelled.is_set():
                raise TimeoutError("rollout batch cancelled")
            if self.connection.poll(min(0.05, max(0., budget.deadline - time.monotonic()))):
                return self.connection.recv()
            if not self.process.is_alive():
                raise RuntimeError(f"rollout worker exited with code {self.process.exitcode}")

    def execute(self, function, args, budget, batch_cancelled):
        try:
            budget.check()
            if batch_cancelled.is_set():
                raise TimeoutError("rollout batch cancelled")
            if self.process is None:
                context = mp.get_context("spawn")
                self.connection, child = context.Pipe()
                self.process = context.Process(target=_worker, args=(child,))
                try:
                    self.process.start()
                finally:
                    child.close()
                status, _ = self.receive(budget, batch_cancelled)
                if status != "ready":
                    raise RuntimeError("rollout worker did not initialize")
            self.connection.send((function, args))
            status, value = self.receive(budget, batch_cancelled)
            if status == "ok":
                return value
            kind, message = value
            if kind == "TimeoutError":
                raise TimeoutError(message)
            raise RuntimeError(f"rollout {kind}: {message}")
        except BaseException:
            self.stop()
            raise


class RolloutPool:
    """One persistent process per active slot, shared across candidate batches."""

    def __init__(self, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rollout-watchdog")
        self.local = threading.local()
        self.lock = threading.Lock()
        self.slots = []
        self.closed = False
        atexit.register(self.shutdown)

    def _execute(self, function, args, budget, cancelled, task_timeout):
        if task_timeout is not None:
            task_budget = EvaluationBudget(min(budget.deadline, time.monotonic() + task_timeout))
            task_budget.cancelled = budget.cancelled
            budget = task_budget
        slot = getattr(self.local, "slot", None)
        if slot is None:
            slot = self.local.slot = _Slot()
            with self.lock:
                self.slots.append(slot)
        return slot.execute(function, args, budget, cancelled)

    def map(self, function, arguments, *, budget=None, task_timeout=None):
        budget = budget or getattr(_local, "budget", None)
        if budget is None:
            raise RuntimeError("rollout requires an evaluation budget")
        if task_timeout is not None and (not math.isfinite(task_timeout) or task_timeout <= 0):
            raise ValueError("task_timeout must be finite and positive")
        budget.check()
        cancelled = threading.Event()
        futures = {}
        try:
            for index, args in enumerate(arguments):
                future = self.executor.submit(self._execute, function, args, budget, cancelled, task_timeout)
                futures[future] = index
            results = [None] * len(futures)
            for future in as_completed(futures, timeout=max(0., budget.deadline - time.monotonic())):
                results[futures[future]] = future.result()
            budget.check()
            return results
        except BaseException as exc:
            cancelled.set()
            for future in futures:
                future.cancel()
            # Active watchdogs kill and reap their own workers before returning.
            wait(futures)
            # Python 3.10 uses a distinct concurrent.futures timeout type.
            if isinstance(exc, (TimeoutError, FuturesTimeoutError)):
                raise TimeoutError("rollout deadline exceeded; workers terminated and reaped") from exc
            raise

    def shutdown(self, wait=True, cancel_futures=True):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=cancel_futures)
        for slot in self.slots:
            slot.stop()
        atexit.unregister(self.shutdown)
