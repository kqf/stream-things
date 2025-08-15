import cv2
import mediapipe as mp
import numpy as np


def extract_eye_patch(
    frame,
    landmarks,
    indices,
    scale=2.5,
    target_size=64,
):
    h, w, _ = frame.shape
    points = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in indices],
        dtype=np.float32,
    )
    cx, cy = points.mean(axis=0)

    eye_w = np.linalg.norm(points[1] - points[3])
    box_size = eye_w * scale

    x1 = int(cx - box_size / 2)
    y1 = int(cy - box_size / 2)
    x2 = int(cx + box_size / 2)
    y2 = int(cy + box_size / 2)

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)

    eye_patch = frame[y1:y2, x1:x2]
    if eye_patch.shape[0] == 0 or eye_patch.shape[1] == 0:
        return frame[:64, :64], (0, 0)

    resized = cv2.resize(eye_patch, (target_size, target_size))
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    input_tensor = normalized[np.newaxis, :, :, :]

    return input_tensor, (x1, y1)


def draw_eye(frame, face_landmarks, indices, infer_iris):
    input_tensor, top_left = extract_eye_patch(frame, face_landmarks, indices)
    if input_tensor is None:
        return

    iris_landmarks = infer_iris(input_tensor)
    x, y, _ = iris_landmarks.reshape(-1, 3)[-1]  # Last point is the center
    x = x / 2.0
    y = y / 2.0

    px = int(x + top_left[0])
    py = int(y + top_left[1])
    print(px, py)
    cv2.circle(frame, (px, py), 2, (0, 0, 255))


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


# Load the symbol mask once globally
SYMBOL_PATH = "pattern.jpg"  # Path to your uploaded swirly pattern
symbol_img = cv2.imread(SYMBOL_PATH, cv2.IMREAD_GRAYSCALE)

# Binarize the symbol to create a mask
_, symbol_mask = cv2.threshold(symbol_img, 128, 255, cv2.THRESH_BINARY)


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
    frame: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    output = frame.copy()
    xmin, ymin, xmax, ymax = bbox

    # Original box dimensions
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return output

    # Largest side length
    side = max(w, h)

    # Center of the original box
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

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
        symbol_mask, (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_AREA
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


def main():
    # Constants
    mp_face_mesh = mp.solutions.face_mesh  # type: ignore
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # Required for iris tracking
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0].landmark
            draw_face_box(frame, face)
            bbox = to_xyxy(frame, face)
            frame = paint_face(frame, bbox)
        cv2.imshow("Iris tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
