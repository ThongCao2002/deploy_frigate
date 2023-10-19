import glob
import time

import cv2
from plate_detector_gpu import Plate_Detector
from retina_plate.utils.utils import img_transform

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    path_folder = "/workspace/frigate/debug/plate_test/*.jpg"
    path_save = "/workspace/frigate/debug/plate_test_out/100img.txt"
    path_image = "/workspace/frigate/debug/plate_test_out/"
    # plate_w = open(path_save,"w+")
    detector = Plate_Detector()
    # recognizer = Plate_Recognizer()
    lst_img = glob.glob(path_folder)
    lst_time = []
    for idx, p_img in enumerate(lst_img):
        print(p_img)

        plate_name = []
        # Load image
        image = cv2.imread(p_img)

        # Detect processing
        tik = time.time()
        image_processed = detector.get_input(image)
        model_output = detector.detect(image_processed)
        detection_result = detector.post_process(
            model_output[0], model_output[1], model_output[2]
        )
        lst_time.append(time.time() - tik)
        print(lst_time[-1])
        print(detection_result)
        if len(detection_result) != 0:
            for i in range(0, len(detection_result)):
                plate = img_transform(image, detection_result[i][5:])
                # cv2.imshow("plate", plate)
                # cv2.waitKey()
                # Do dilate to cut image into 2 lines (need to know if image is one or two lines)

                # ocr_image_processed = recognizer.get_input(plate)
                # ocr_model_out = recognizer.run(ocr_image_processed)
                # ocr_result = recognizer.post_process(ocr_model_out)
                # plate_name.append(ocr_result[0])
            for i, b in enumerate(detection_result):
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
                cx = b[0]
                cy = b[1] + 12
                # plate_w.write( plate_name[i] +'\n')
                # cv2.putText(
                #     image,
                #     plate_name[i],
                #     (cx, cy),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0, 255, 0),
                # )
                # landms
                cv2.circle(image, (b[5], b[6]), 5, (0, 0, 255), 1)  # red, top left
                cv2.circle(
                    image, (b[7], b[8]), 5, (0, 255, 255), 1
                )  # yellow, top right
                cv2.circle(
                    image, (b[9], b[10]), 5, (255, 0, 255), 1
                )  # purple, bottom left
                cv2.circle(
                    image, (b[11], b[12]), 5, (0, 255, 0), 1
                )  # green, bottom right

            # cv2.imshow("Test", cv2.resize(image, (1600, 900)))
            # cv2.imshow("plate", plate)
            cv2.imwrite(path_image + str(idx) + ".jpg", image)
            # cv2.waitKey()
    # plate_w.close()

    # print("process time:", sum(lst_time)/len(lst_time))
    # print("done")
    # print("process time:", sum(lst_time)/len(lst_time))
    # print("done")
