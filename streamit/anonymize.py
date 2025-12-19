from contextlib import contextmanager

import cv2
import mediapipe as mp
import numpy as np

XYWH = tuple[int, int, int, int]


def draw_face_box(frame, landmarks, color=(255, 0, 0)):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def to_xyxy(frame, landmarks):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return x1, y1, x2, y2


def read_the_pattern(path):
    symbol_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, symbol_mask = cv2.threshold(symbol_img, 128, 255, cv2.THRESH_BINARY)
    return symbol_mask


def swirl_effect(img, strength=5, radius=200):
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Mesh grid of coordinates
    y, x = np.indices((h, w), dtype=np.float32)
    x_c = x - center_x
    y_c = y - center_y
    r = np.sqrt(x_c**2 + y_c**2)
    theta = np.arctan2(y_c, x_c) + strength * np.exp(-r / radius)

    map_x = (r * np.cos(theta) + center_x).astype(np.float32)
    map_y = (r * np.sin(theta) + center_y).astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def to_mask(img_gray_3ch, threshold=1):
    # img_gray_3ch is a 3-channel grayscale image (0-255)
    # We'll convert it to single channel by taking one channel (all are equal)
    gray_single = img_gray_3ch[..., 0]
    # Create a binary mask where pixels > threshold are 1, else 0
    mask = (gray_single > threshold).astype(np.float32)
    # Expand dims back to 3 channels for masking
    return np.stack([mask] * 3, axis=-1)


def paint_face(
    frame: np.ndarray,
    bbox: XYWH,
    pattern: np.ndarray,
) -> np.ndarray:
    output = frame.copy()
    xmin, ymin, w, h = bbox

    # Original box dimensions
    if w <= 0 or h <= 0:
        return output

    # Largest side length
    side = max(w, h)

    # Center of the original box
    cx = xmin + w // 2
    cy = ymin + h // 2 - h // 12

    # New square coordinates
    xmin = cx - side // 2
    ymin = cy - side // 2
    xmax = xmin + side
    ymax = ymin + side

    # Clamp coordinates to image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(frame.shape[1], xmax)
    ymax = min(frame.shape[0], ymax)

    # Extract ROI
    roi = output[ymin:ymax, xmin:xmax]

    # Apply swirl effect (two variations for complexity)
    swirl1 = swirl_effect(roi, strength=4, radius=min(roi.shape[:2]) // 2)
    swirl2 = swirl_effect(roi, strength=5, radius=min(roi.shape[:2]))

    # Convert both swirls to grayscale, then back to 3 channels
    swirl1_gray = cv2.cvtColor(swirl1, cv2.COLOR_BGR2GRAY)
    swirl2_gray = cv2.cvtColor(swirl2, cv2.COLOR_BGR2GRAY)
    swirl1_gray = cv2.merge([swirl1_gray] * 3)
    swirl2_gray = cv2.merge([swirl2_gray] * 3)

    # Resize mask to square size
    mask_resized = cv2.resize(
        pattern, (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_AREA
    )
    mask_3ch = cv2.merge([mask_resized] * 3)
    mask1_float = mask_3ch.astype(np.float32) / 255.0
    # to_mask -- we consider everything a pattern that above threshold
    mask2_float = to_mask(swirl2_gray, 100)

    # Blend grayscale swirls with original ROI
    blended = (
        (
            roi.astype(np.float32)
            - swirl2_gray.astype(np.float32) * mask1_float
            + swirl1_gray.astype(np.float32) * mask2_float
        )
        .clip(0, 255.0)
        .astype(np.uint8)
    )

    output[ymin:ymax, xmin:xmax] = blended
    return output


def draw_bbox_xywh(frame, bbox, color=(0, 255, 0), thickness=2):
    if bbox is None:
        return frame

    x, y, w, h = map(int, bbox)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame


def head_bbox_from_pose(frame, head_landmarks, padding=0.2):
    h, w, _ = frame.shape

    xs = [int(lm.x * w) for lm in head_landmarks if lm.visibility > 0.5]
    ys = [int(lm.y * h) for lm in head_landmarks if lm.visibility > 0.5]

    if not xs or not ys:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    h = (y2 - y1) * 6
    return (x1, y1 - h // 3, x2 - x1, h)  # XYWH


@contextmanager
def extract_position():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    def extract(frame: np.ndarray) -> XYWH | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks.landmark

        head_landmarks = [
            landmarks[mp_pose.PoseLandmark.NOSE],
            landmarks[mp_pose.PoseLandmark.LEFT_EYE],
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE],
            landmarks[mp_pose.PoseLandmark.LEFT_EAR],
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR],
        ]

        return head_bbox_from_pose(frame, head_landmarks)

    yield extract


def main():
    # Constants
    cap = cv2.VideoCapture(0)
    with extract_position() as extract:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.imread("man-standing.avif")
            pattern = np.ones((512, 512, 1), dtype=np.uint8)
            bbox = extract(frame)
            frame = draw_bbox_xywh(frame, bbox)
            frame = paint_face(frame, bbox, pattern=pattern)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
