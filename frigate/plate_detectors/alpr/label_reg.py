import random

import cv2
import glob
import os
from plate_detector import Plate_Detector
from plate_recognizer import Plate_Recognizer
from retina_plate.utils.utils import img_transform
import time
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path_direc = "/home/can/AI_Camera/FR_Occlusion/Dữ liệu biển số/62k_trainOCR_processed/20000"
    path_save = "/home/can/AI_Camera/FR_Occlusion/Dữ liệu biển số/62k_trainOCR/label.txt"
    path_save_img = "/home/can/AI_Camera/FR_Occlusion/Dữ liệu biển số/62k_trainOCR_processed/20000_model1"

    lst_folder = os.listdir(path_direc)

    plate_w = open(path_save,"w+")
    # detector = Plate_Detector()
    recognizer = Plate_Recognizer()
    count = 0
    for f in lst_folder:
        path_folder = os.path.join(path_direc, f)
        # lst_img = glob.glob(path_folder)
    #     random.shuffle(lst_img)
    #     lst_time = []
    #     for idx, p_img in enumerate(lst_img):
    #         print("\n ***************")
    #         print(count)
    #         count += 1
    #         print(p_img)
    #         s_img = os.path.join(path_save_img, f)
    #         if not os.path.exists(s_img):
    #             os.mkdir(s_img)
    #
    #         name_img = p_img.split("/")[-1]
    #         plate_name = []
    #         # Load image
    #         image = cv2.imread(p_img)
    #
    #         # Detect processing
    #         tik = time.time()
    #         image_processed = detector.get_input(image)
    #         model_output = detector.detect(image_processed)
    #         detection_result = detector.post_process(model_output[0], model_output[1], model_output[2])
    #         lst_time.append(time.time()-tik)
    #         print(detection_result)
    #         if len(detection_result) !=0:
    #             for i in range(0, len(detection_result)):
    #                 plate = img_transform(image, detection_result[i][5:])
    #                 cv2.imwrite(s_img+"/"+name_img, plate)
    #                 # cv2.imshow("plate", plate)
    #                 # cv2.waitKey()
    #                 # Do dilate to cut image into 2 lines (need to know if image is one or two lines)
    #
    #                 ocr_image_processed = recognizer.get_input(plate)
    #                 ocr_model_out = recognizer.run(ocr_image_processed)
    #                 ocr_result = recognizer.post_process(ocr_model_out)
    #                 plate_name.append(ocr_result[0])
    #             for i, b in enumerate(detection_result):
    #                 # text = "{:.4f}".format(b[4])
    #                 # b = list(map(int, b))
    #                 # cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
    #                 # cx = b[0]
    #                 # cy = b[1] + 12
    #                 plate_w.write(f + "/" + name_img + " " + plate_name[i] +'\n')
    #             #     cv2.putText(image, plate_name[i], (cx, cy),
    #             #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
    #             #     # landms
    #             #     cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 1)  # red, top left
    #             #     cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 1)  # yellow, top right
    #             #     cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 1)  # purple, bottom left
    #             #     cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 1)  # green, bottom right
    #
    #             # # cv2.imshow('Test', image)
    #             # # cv2.imshow("plate", plate)
    #             # cv2.imwrite(path_image+str(idx)+".jpg", image)
    #             # cv2.waitKey()
    #         # if idx == 4:
    #         #     # plate_w.close()
    #         #     break
    #
    # plate_w.close()

    # print("process time:", sum(lst_time)/len(lst_time))
    # print("done")


        # name_img = p_img.split("/")[-1]
        # plate_name = []
        # Load image
        image = cv2.imread(path_folder)
        ocr_image_processed = recognizer.get_input(image)
        ocr_model_out = recognizer.run(ocr_image_processed)
        ocr_result = recognizer.post_process(ocr_model_out)
        image_name = os.path.join(path_save_img, ocr_result[0] + "_" + str(count) + ".jpg")
        # cv2.imshow("Hinh", image)
        cv2.imwrite(image_name, image)
        print(ocr_result)
        count += 1
        # cv2.waitKey(0)