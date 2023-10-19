import cv2
import numpy as np


def img_transform(img, point, ratio=0.33):
    """ """
    point = [int(x) for x in point]
    xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr = point

    rect = np.array([[xtl, ytl], [xtr, ytr], [xbr, ybr], [xbl, ybl]], dtype="float32")

    widthA = np.sqrt(((xbr - xbl) ** 2) + ((ybr - ybl) ** 2))
    widthB = np.sqrt(((xtr - xtl) ** 2) + ((ytr - ytl) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((xtr - xbr) ** 2) + ((ytr - ybr) ** 2))
    heightB = np.sqrt(((xtl - xbl) ** 2) + ((ytl - ybl) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    img_warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), borderValue=0)

    height, width, channels = img_warped.shape
    if height % 2 != 0:
        height -= 1
    if height / width > ratio:
        croppedImage_top = img_warped[0 : int(height / 2), 0:width]  # this line crops
        croppedImage_bot = img_warped[
            int(height / 2) : height, 0:width
        ]  # this line crops
        plate = cv2.hconcat([croppedImage_top, croppedImage_bot])
        return True, plate
    else:
        return False, img_warped


def draw_result(image, detection_result, ocr_result, tracking_id, color=(0, 0, 255)):
    # Printing stuff
    b = list(map(int, detection_result))
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 1)
    cx = b[0]
    cy = b[1] + 25
    text_size, _ = cv2.getTextSize(ocr_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    text_w, text_h = text_size
    cv2.rectangle(image, (b[0], b[1]), (cx + text_w, cy + text_h - 5), (0, 0, 0), -1)
    cv2.putText(
        image,
        ocr_result,
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        thickness=2,
    )
    # cv2.putText(image, str(tracking_id), (cx, cy+20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

    # landms
    cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 1)  # red, top left
    cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 1)  # yellow, top right
    cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 1)  # purple, bottom left
    cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 1)  # green, bottom right
    return image
