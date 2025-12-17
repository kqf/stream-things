import cv2
import numpy as np
import pytest

from streamit.anonymize import draw_bbox_xywh, paint_face


@pytest.fixture
def frame() -> np.ndarray:
    return cv2.imread("man-standing.jpg")


@pytest.fixture
def pattern(h=512, w=512, c=1) -> np.ndarray:
    noise = np.random.randint(0, 255, (h * w * c))
    return noise.reshape(h, w, c).astype(np.uint8) * 0 + 255.0


def test_paints_face(frame, pattern):
    bbox = (207, 28, 70, 88)
    frame = draw_bbox_xywh(frame, (207, 28, 70, 88))
    print(pattern.shape)
    frame = paint_face(frame, bbox, pattern=pattern)
    cv2.imshow("frame", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
