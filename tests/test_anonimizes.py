import cv2
import numpy as np
import pytest

from streamit.anonymize import draw_bbox_xywh


@pytest.fixture
def frame() -> np.ndarray:
    return cv2.imread("man-standing.jpg")


def test_paints_face(frame):
    frame = draw_bbox_xywh(frame, (207, 28, 70, 88))
    # frame = paint_face(frame, bbox)
    cv2.imshow("frame", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
