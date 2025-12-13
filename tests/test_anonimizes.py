import cv2

from streamit.anonymize import draw_bbox_xywh


def test_paints_face(frame):
    frame = cv2.imread("man-standing.avif")
    frame = draw_bbox_xywh(frame, (328, 80, 94, 138))
    # frame = paint_face(frame, bbox)
    cv2.imshow("frame", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
