import cv2

# === Configuration ===

rtmp_url = "rtmp://192.168.0.11:1935/stream/hello"

SKIP_FRAMES = 30
OUTPUT_FPS = 15
OUTPUT_FILENAME = "timelapse.mp4"
VIDEO_CODEC = "mp4v"

# Desired recording resolution (width, height)
# RECORD_RESOLUTION = 848 // 2, 480 // 2  # (480, 360)
RECORD_RESOLUTION = None
# RECORD_RESOLUTION = None  # Use None to keep original


def build_writer(resolution):
    return cv2.VideoWriter(
        OUTPUT_FILENAME,
        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
        OUTPUT_FPS,
        resolution,
    )


def adjust_videowriter(frame, requested_resolution, current_writer):
    height, width, _ = frame.shape
    print(f"📐 Stream resolution: {width}x{height}")

    if requested_resolution is None:
        return build_writer((width, height))

    req_w, req_h = requested_resolution
    if req_w >= width or req_h >= height:
        print("ℹ️ Requested resolution >= stream — using original resolution.")
        return current_writer

    final_resolution = (req_w, req_h)
    print(f"📉 Resizing recorded frames to {req_w}x{req_h}")

    current_writer.release()
    return build_writer(final_resolution)


def resize_frame(frame, target_resolution):
    if target_resolution is None:
        return frame

    height, width, _ = frame.shape
    target_w, target_h = target_resolution

    if target_w >= width or target_h >= height:
        return frame

    return cv2.resize(
        frame, (target_w, target_h), interpolation=cv2.INTER_AREA
    )


def record_timelapse(cap, video_writer, target_resolution):
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Stream ended or failed — finishing up...")
            break

        resized_frame = resize_frame(frame, target_resolution)
        cv2.imshow("Timelapse Capture", resized_frame)  # Show original stream
        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("🛑 'q' pressed — exiting...")
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        video_writer.write(resized_frame)
        # cv2.resizeWindow("Timelapse Capture", 848 // 2, 480 // 2)


def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Error: Cannot open RTMP stream")
        return

    print("✅ RTMP stream opened successfully")
    writer = build_writer(RECORD_RESOLUTION)

    try:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read first frame from stream")
            return

        writer = adjust_videowriter(frame, RECORD_RESOLUTION, writer)

        print(
            f"🎥 Recording timelapse to '{OUTPUT_FILENAME}' "
            f"at {OUTPUT_FPS} FPS, "
            f"skipping every {SKIP_FRAMES} frames"
        )

        record_timelapse(cap, writer, RECORD_RESOLUTION)

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"✅ Timelapse saved as '{OUTPUT_FILENAME}'")


if __name__ == "__main__":
    main()
