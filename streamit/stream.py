import logging
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@contextmanager
def dynamic_writer(filename: Path, codec: str = "mp4v", fps: int = 30):
    writer = None

    def write(frame: np.ndarray):
        nonlocal writer
        if not writer:
            h, w, *_ = frame.shape
            writer = cv2.VideoWriter(
                str(filename),
                cv2.VideoWriter_fourcc(*codec),  # type: ignore
                fps,
                (w, h),
            )
        writer.write(frame)

    yield write

    if writer is not None:
        writer.release()


def streamopen(stream=0):
    capture = cv2.VideoCapture(stream)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        yield frame
    capture.release()


def resize(frame, height=None):
    if height is None:
        return frame

    h, w, _ = frame.shape

    if height >= h:
        return frame

    scale = height / h
    new_w = int(w * scale)

    return cv2.resize(
        frame,
        (new_w, height),
        interpolation=cv2.INTER_AREA,
    )


def main(h: int | None = None, skip: int = 30) -> None:
    with dynamic_writer(Path("test.mp4")) as write:
        for frame_count, frame in enumerate(streamopen()):
            resized_frame = resize(frame, h)
            cv2.imshow("Timelapse Capture", resized_frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                logger.info("'q' pressed — exiting...")
                break

            frame_count += 1
            if frame_count % skip != 0:
                continue

            write(resized_frame)


if __name__ == "__main__":
    main()
