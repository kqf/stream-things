from dataclasses import dataclass, field
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


def build_writer(resolution: tuple[int, int]) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        OUTPUT_FILENAME,
        cv2.VideoWriter_fourcc(*VIDEO_CODEC),  # type: ignore
        OUTPUT_FPS,
        resolution,
    )


@dataclass
class DynamicWriter:
    filename: Path
    codec: str
    fps: int
    writer: cv2.VideoWriter | None = field(init=False)

    def write(self, frame: np.ndarray) -> None:
        if not self.writer:
            h, w, *_ = frame.shape
            self.writer = build_writer((h, w))
        self.writer.write(frame)


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
        cv2.imshow("Timelapse Capture", resized_frame)  # Show original stream
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
    writer = build_writer(RECORD_RESOLUTION)
    print(
        f"🎥 Recording timelapse to '{OUTPUT_FILENAME}' "
        f"at {OUTPUT_FPS} FPS, "
        f"skipping every {SKIP_FRAMES} frames"
    )
    record_timelapse(cap, writer, RECORD_RESOLUTION)
    print(f"✅ Timelapse saved as '{OUTPUT_FILENAME}'")


if __name__ == "__main__":
    main()
