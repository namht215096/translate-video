"""GUI-side handle on the engine process (no torch import here, so the window never blocks on CUDA)."""

import multiprocessing as mp
import queue

from .engine import Settings, engine_main
from .windows_env import PROJECT_ROOT


class EngineClient:
    def __init__(self, root=PROJECT_ROOT):
        self.root = root
        self.context = mp.get_context("spawn")
        self.process = None
        self.commands = self.events = None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def ensure_running(self):
        if self.alive:
            return
        self.commands, self.events = self.context.Queue(), self.context.Queue()
        self.process = self.context.Process(
            target=engine_main, args=(self.commands, self.events, str(self.root)), daemon=True)
        self.process.start()

    def send(self, *command):
        self.commands.put(command)

    def poll(self, limit=200):
        """Pending events, newest last. Empty when the engine has not started."""
        out = []
        while self.events is not None and len(out) < limit:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out

    def shutdown(self, timeout=10.0):
        if not self.alive:
            return
        self.send("shutdown")
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)


__all__ = ["EngineClient", "Settings"]
