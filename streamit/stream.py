from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

# === Configuration ===

rtmp_url = "rtmp://192.168.0.11:1935/stream/hello"

SKIP_FRAMES = 30
OUTPUT_FPS = 15
OUTPUT_FILENAME = "timelapse.mp4"
VIDEO_CODEC = "mp4v"

# Desired recording resolution (width, height)
RECORD_RESOLUTION = None


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


def adjust_resolution(frame, requested_resolution):
    height, width, _ = frame.shape
    print(f"📐 Stream resolution: {width}x{height}")

    if requested_resolution is None:
        return width, height

    req_w, req_h = requested_resolution
    if req_w >= width or req_h >= height:
        return requested_resolution

    return req_w, req_h


def resize_frame(frame, target_resolution):
    if target_resolution is None:
        return frame

    height, width, _ = frame.shape
    target_w, target_h = target_resolution

    if target_w >= width or target_h >= height:
        return frame

    return cv2.resize(
        frame,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )


def record_timelapse(cap, target_resolution):
    frame_count = 0

    with dynamic_writer(Path("test.mp4")) as write:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resolution = adjust_resolution(frame, target_resolution)
            resized_frame = resize_frame(frame, resolution)
            cv2.imshow("Timelapse Capture", resized_frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("🛑 'q' pressed — exiting...")
                break

            frame_count += 1
            if frame_count % SKIP_FRAMES != 0:
                continue

            write(resized_frame)


def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("✅ RTMP stream opened successfully")
    print(
        f"🎥 Recording timelapse to '{OUTPUT_FILENAME}' "
        f"at {OUTPUT_FPS} FPS, "
        f"skipping every {SKIP_FRAMES} frames"
    )
    record_timelapse(cap, RECORD_RESOLUTION)
    print(f"✅ Timelapse saved as '{OUTPUT_FILENAME}'")


if __name__ == "__main__":
    main()
