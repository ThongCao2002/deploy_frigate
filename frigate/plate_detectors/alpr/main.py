import os

import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from plate_checker import Plate_Checker
from plate_detector_gpu import Plate_Detector
from plate_recognizer import Plate_Recognizer
from retina_plate.utils.utils import img_transform

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    detector = Plate_Detector()
    recognizer = Plate_Recognizer()
    checker = Plate_Checker()
    plate_name = []
    # Load image
    image = cv2.imread("test.jpg")

    # Detect processing
    image_processed = detector.get_input(image)
    model_output = detector.detect(image_processed)
    detection_result = detector.post_process(
        model_output[0], model_output[1], model_output[2]
    )
    print(detection_result)
    for i in range(0, len(detection_result)):
        _, plate = img_transform(image, detection_result[i][5:])
        # cv2.imshow("plate", plate)
        # cv2.waitKey()
        # Do dilate to cut image into 2 lines (need to know if image is one or two lines)

        ocr_image_processed = recognizer.get_input(plate)
        ocr_model_out = recognizer.run(ocr_image_processed)
        ocr_result = recognizer.post_process(ocr_model_out)
        plate_name.append(ocr_result[0])
    for i, b in enumerate(detection_result):
        if not checker.run(plate_name[i]):
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        cx = b[0]
        cy = b[1] + 24
        cv2.putText(
            image, plate_name[i], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)
        )
        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 1)  # red, top left
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 1)  # yellow, top right
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 1)  # purple, bottom left
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 1)  # green, bottom right

    # cv2.imshow('Test', image)
    # cv2.imshow("plate", plate)
    # cv2.waitKey()
    # cv2.waitKey()
