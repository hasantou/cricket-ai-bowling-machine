"""
The missing piece between the already-real pretrained pose model
(pose_estimation.py) and an actual camera mounted on the machine, instead
of a pre-recorded file someone uploads afterward.

Nothing about PoseEstimator itself needed to change for this: its
extract_landmarks_from_frames() already supports being called repeatedly
in sequence (its own internal timestamp tracking exists specifically so
one instance can keep processing frames indefinitely without restarting —
see that module's docstring). The gap was never the pose model, it was a
live frame source to feed it from, one frame at a time as it arrives,
instead of a whole pre-loaded list. This is that source.

No real camera exists yet to test against. LiveVideoSource wraps
cv2.VideoCapture opened against a device index or stream URL for real
use, or accepts an injected frame generator for tests/demos — the same
dependency-injection pattern SerialMachineController's connection_factory
and ReleaseSensor's simulated stand-in already use elsewhere in this
project. A pre-recorded file also works as a `device` argument (that's
exactly what cv2.VideoCapture does), which is what lets this be honestly
validated against real footage today, "playing it back live," before any
real camera exists — see demo_live_delivery_detection.py.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, Optional

import numpy as np


class LiveVideoSource:
    """Frame source abstraction: a live camera (or a file played back as
    if it were live) is read the same way cv2.VideoCapture always reads
    anything — frame by frame until the source stops. That's the whole
    reason this is straightforward: OpenCV never distinguished "file" from
    "camera" at the read() level, only at how VideoCapture was opened.
    """

    def __init__(
        self,
        device=None,
        fps_hint: float = 30.0,
        frame_factory: Optional[Callable[[], Iterable[np.ndarray]]] = None,
    ):
        """`device`: a camera index (e.g. 0), a stream URL, or a file
        path — passed straight to cv2.VideoCapture. `frame_factory`, if
        given, is a zero-arg callable returning an iterable of frames
        (e.g. a test's synthetic generator) — no real capture device is
        opened at all in that case, used for testing/demoing this module
        without hardware. Exactly one of `device`/`frame_factory` should
        be given.
        """
        self._external_frames = None
        self._cap = None
        if frame_factory is not None:
            self._external_frames = frame_factory
            self.fps = fps_hint
        else:
            import cv2
            self._cap = cv2.VideoCapture(device)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open camera/stream/file: {device!r}")
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or fps_hint

    def frames(self) -> Iterator[np.ndarray]:
        """Yields frames as they arrive: indefinitely for a live camera,
        or until exhausted for a file/injected source. A caller looping
        over this naturally gets "process as fast as frames arrive"
        behaviour with a real camera — no separate real-time pacing logic
        needed here; that's cv2.VideoCapture.read()'s own blocking
        behaviour against a live device."""
        if self._external_frames is not None:
            yield from self._external_frames()
            return
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "LiveVideoSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
