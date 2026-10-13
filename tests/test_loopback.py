import threading

import numpy as np

from translate_video.loopback import PlaybackQueue


def test_render_pads_with_silence_and_keeps_order():
    queue = PlaybackQueue()
    queue.push(np.arange(1, 6, dtype=np.float32))
    assert queue.render(3).tolist() == [1, 2, 3]
    assert queue.pending_frames() == 2
    assert queue.render(4).tolist() == [4, 5, 0, 0]
    assert queue.pending_frames() == 0


def test_wait_drained_returns_after_render_and_on_stop():
    queue = PlaybackQueue()
    stopped = threading.Event()
    queue.push(np.ones(4, np.float32))
    waiter = threading.Thread(target=queue.wait_drained, args=(stopped,))
    waiter.start()
    queue.render(4)
    waiter.join(timeout=2)
    assert not waiter.is_alive()
    queue.push(np.ones(4, np.float32))
    stopped.set()
    queue.wait_drained(stopped)  # must not block once stopped


def test_clear_discards_pending_audio():
    queue = PlaybackQueue()
    queue.push(np.ones(10, np.float32))
    queue.clear()
    assert queue.render(2).tolist() == [0, 0]
