import math

import cv2
import time
import random
import numpy as np
import os
from plate_detector_gpu import Plate_Detector
from plate_recognizer import Plate_Recognizer
from plate_checker import Plate_Checker
from retina_plate.utils.utils import img_transform, draw_result
from plate_tracker import Sort


class AutoLPR:
    def __init__(self, gpu_id=0):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.detector = Plate_Detector()
        self.recognizer = Plate_Recognizer()
        self.checker = Plate_Checker()
        self.tracker = Sort()
        self.fps = 0
        self.image_size = (1366, 768)
        self.working_zone = [50, 50, 1300, 700]
        self.current_plate_number = 'io'
        self.tracking_id = float("inf")
        self.is_2line, self.license_plate_format = True, 0

    def processFrame(self, input_frame):

        image = cv2.resize(input_frame, self.image_size, interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(image, (self.working_zone[0], self.working_zone[1]), (self.working_zone[2], self.working_zone[3]),
                      (0, 255, 0), 1)
        # Get plate bounding boxes
        start_time = time.time()
        image_processed = self.detector.get_input(image)
        model_output = self.detector.detect(image_processed)
        detection_result = self.detector.post_process(model_output[0], model_output[1], model_output[2])
        print("Detect: ", time.time() - start_time)
        rand = random.randint(0, 10)
        if rand < 3:
            self.fps = int(1.0 / (time.time() - start_time))
        cv2.putText(image, str(self.fps), (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # print("Detection time: ", time.time() - start_time)
        # Check if having no license plate bounding detected
        if len(detection_result) == 0:
            predict = self.tracker.update(np.array([]))
            return image
        # Run for any license plate, if biggest box is not plate format then use next bigger one bounding box
        while len(detection_result) != 0:
            # Get largest bounding box
            best_index = self.select_significant_face(detection_result)
            plate_bbox = detection_result[best_index]
            image = draw_result(image, plate_bbox, self.current_plate_number, self.tracking_id, color=(0, 255, 255))
            # Tracking plate
            tracking_time = time.time()
            # print("Detect: ", plate_bbox[:5])
            predict = self.tracker.update(np.array([plate_bbox[:5]]))
            # print("Tracking time: ", time.time() - tracking_time)
            # print("\n ***********************")
            # print("predict: ", predict)
            if len(predict) == 0:
                # print("mark")
                self.tracking_id = float("inf")
                self.current_plate_number = 'io'
                # detection_result = np.delete(detection_result, best_index, axis=0)
                return image
            id = predict[0][4]
            # print("self ID: ", self.tracking_id)
            # print("tracking ID: ", id)
            # Check tracking ID, if it doesn't change then it is new license plate.
            if id == self.tracking_id:
                # print("mark 2")
                # print("current OCR ", self.current_plate_number)
                if self.current_plate_number == 'io':
                    # print("Getting new OCR")
                    if not self.check_zone(plate_bbox[:4]):
                        # print('ID Out zone', self.tracking_id)
                        # image = draw_result(image, plate_bbox, self.current_plate_number, id)
                        detection_result = np.delete(detection_result, best_index, axis=0)
                        continue
                    else:
                        ocr_time = time.time()
                        plate = img_transform(image, plate_bbox[5:])
                        ocr_image_processed = self.recognizer.get_input(plate)
                        ocr_model_out = self.recognizer.run(ocr_image_processed)
                        ocr_result = self.recognizer.post_process(ocr_model_out)[0]
                        print("OCR: ", time.time() - ocr_time)
                        if not self.checker.run(ocr_result):
                            self.current_plate_number = "io"
                            # print("Check format Fail")
                            # Delete this wrong format plate and check the next one
                            detection_result = np.delete(detection_result, best_index, axis=0)
                            continue
                        self.current_plate_number = ocr_result
                        # print("OCR time: ", time.time() - ocr_time)
                image = draw_result(image, plate_bbox, self.current_plate_number, self.tracking_id)
                # print("FULL FRAME WITH TRACKING: ", time.time() - start_time)
                return image
            else:
                self.tracking_id = id
                # plate_bbox[0:4] = predict[0][:4]
                # Check if box inside working zone, then return non-ocr detect results if outside
                if not self.check_zone(plate_bbox[:4]):
                    # print('ID Out zone new ID', self.tracking_id)
                    image = draw_result(image, plate_bbox, self.current_plate_number, id)
                    return image
                ocr_time = time.time()
                plate = img_transform(image, plate_bbox[5:])
                ocr_image_processed = self.recognizer.get_input(plate)
                ocr_model_out = self.recognizer.run(ocr_image_processed)
                ocr_result = self.recognizer.post_process(ocr_model_out)[0]
                print("OCR time: ", time.time() - ocr_time)
                # Re-check ocr result to know if it good tracking
                # Allow to differ up to 2 characters
                # print(" OCR: ", ocr_result)

                # if editdistance.eval(self.current_plate_number,
                #                      ocr_result) > 2 and self.current_plate_number != 'io':
                #     self.tracking_id = float("inf")
                #     detection_result = np.delete(detection_result, best_index, axis=0)
                #     continue
                # Just printing things to check fps

                # Check plate number's format
                if not self.checker.run(ocr_result):
                    self.current_plate_number = "io"
                    # print("Check format Fail")
                    # Delete this wrong format plate and check the next one
                    detection_result = np.delete(detection_result, best_index, axis=0)
                    continue
                self.current_plate_number = ocr_result
                # print("Current OCR", self.current_plate_number)
                image = draw_result(image, plate_bbox, self.current_plate_number, self.tracking_id)
                # print("FULL FRAME NO TRACKING: ", time.time() - start_time)
                return image
        return image

    def predict(self, image, background_image=None, height_ratio=1, width_ratio=1):
        """
        Get license plate locations and OCR value. It'll return the biggest license plate value, even having many
        license plates.
        :param image: opencv format image. (B, G, R)
        :param background_image: motion/background image. Binary image
        :param height_ratio: (0, 1). Want-to-keep ratio of cropping height dimension of input image. 1 to keep the original image.
        :param width_ratio:(0, 1). Want-to-keep ratio of cropping weight dimension of input image. 1 to keep the original image.
        :return: (x_min_box, y_min_box, x_max_box, y_max_box, 5 x (x_landmark, y_landmark), ocr value)
        """
        # Get plate bounding boxes
        start_time = time.time()

        # Cropping image
        height, width, channels = image.shape
        # print(image.shape)
        x_new = int((1 - width_ratio) * (width / 2))
        y_new = int((1 - height_ratio) * (height / 2))
        image = image[y_new: height - y_new, x_new: width - x_new]
        # Update working zone
        min_x = int(image.shape[0] / 12)
        self.working_zone = [min_x, min_x,
                             image.shape[1] - min_x, image.shape[0] - min_x]
        image_processed = self.detector.get_input(image)
        model_output = self.detector.detect(image_processed)
        detection_result = self.detector.post_process(model_output[0], model_output[1], model_output[2])
        # print("--------------------------------------")
        # print("Detect {} license plate in {}s ".format(len(detection_result), time.time() - start_time))
        # Run for any license plate, if biggest box is not plate format then use next bigger one bounding box
        while len(detection_result) != 0:
            # Get largest bounding box
            best_index = self.select_significant_face(detection_result)
            plate_bbox = detection_result[best_index]
            # Check zone
            if not self.check_zone(plate_bbox[:4]):
                print("Out Zone")
                detection_result = np.delete(detection_result, best_index, axis=0)
                continue
            # Check motion
            if background_image is not None:
                original_image_box = plate_bbox.copy()
                if not self.check_motion(original_image_box[:4], background_image):
                    detection_result = np.delete(detection_result, best_index, axis=0)
                    continue
            ocr_time = time.time()
            self.is_2line, plate = img_transform(image, plate_bbox[5:])
            ocr_image_processed = self.recognizer.get_input(plate)
            ocr_model_out = self.recognizer.run(ocr_image_processed)
            ocr_result = self.recognizer.post_process(ocr_model_out)[0]
            # print("OCR time: ", time.time() - ocr_time)
            # Check plate number's format
            self.license_plate_format, is_valid_format = self.checker.run(ocr_result)
            if not is_valid_format:
                # Delete this wrong format plate and check the next one
                detection_result = np.delete(detection_result, best_index, axis=0)
                continue
            # print("Current OCR", ocr_result)
            plate_bbox[:4][::2] += x_new
            plate_bbox[:4][1::2] += y_new
            plate_bbox[5:][::2] += x_new
            plate_bbox[5:][1::2] += y_new
            output = plate_bbox.tolist()
            output.extend([ocr_result])
            return output
        return None

    def check_zone(self, plate_box, overlap_threshold=0.75):
        # determine the coordinates of the intersection rectangle
        x_left = max(plate_box[0], self.working_zone[0])
        y_top = max(plate_box[1], self.working_zone[1])
        x_right = min(plate_box[2], self.working_zone[2])
        y_bottom = min(plate_box[3], self.working_zone[3])

        if x_right < x_left or y_bottom < y_top:
            return False
        if plate_box[2] < plate_box[0] or plate_box[3] < plate_box[1]:
            return False
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both plate bounding box
        plate_area = (plate_box[2] - plate_box[0]) * (plate_box[3] - plate_box[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / plate_area
        assert iou >= 0.0
        assert iou <= 1.0
        return iou > overlap_threshold

    def check_motion(self, box, background, image=None, threshold_motion=0.3):
        # Background image is having difference size with scale of bounding box
        # Scale box coordinate to fix with background image
        bg_height, bg_width = background.shape
        scale = float(bg_width) / float(self.detector.origin_size[1])
        scaled_box = box * scale
        scaled_box = [int(x) for x in scaled_box]
        crop_image = background[scaled_box[1]: scaled_box[3], scaled_box[0]: scaled_box[2]]
        if crop_image.shape[0] == 0 or crop_image.shape[1] == 0:
            print("Negative bounding box!")
            return False
        white_count = cv2.countNonZero(crop_image)
        motionBox_rate = white_count / (crop_image.shape[0] * crop_image.shape[1])
        if image is not None:
            # debug
            img_debug = background.copy()
            img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
            scale_debug = float(bg_width) / float(image.shape[1])
            image = cv2.resize(image, None, None, fx=scale_debug, fy=scale_debug, interpolation=cv2.INTER_LINEAR)
            img_debug = cv2.bitwise_or(image, img_debug)
            if motionBox_rate > threshold_motion:
                cv2.rectangle(img_debug, (scaled_box[0], scaled_box[1]), (scaled_box[2], scaled_box[3]), (255, 0, 0), 2)
            else:
                cv2.rectangle(img_debug, (scaled_box[0], scaled_box[1]), (scaled_box[2], scaled_box[3]), (0, 0, 255), 2)
            cv2.imshow("test", img_debug)

        # cv2.waitKey()

        if motionBox_rate > threshold_motion:
            return True
        else:
            print("Box is not in Motion")
            return False

    @staticmethod
    def select_significant_face(bounding_boxes):
        best_index = 0
        best_area = 0
        for i, b in enumerate(bounding_boxes):
            bbox_w, bbox_h = b[2] - b[0], b[3] - b[1]
            area = bbox_w * bbox_h
            if best_area <= area:  # rank > best_rank and
                best_area = area
                best_index = i
        return best_index

    def post_process(self, ocr_result):
        """
            Convert license plate number from non-symbol to symbol
            :param ocr_result: recognized license plate number
            :param license_plate_format:
            :param is_2line: True if licence plate has 2 line
            :return: licence plate number with symbol
        """
        temp_text = ''
        if self.is_2line:
            if self.license_plate_format == 1:
                # bien dan dung 2 dong
                if ocr_result[:2].isdigit():
                    temp_text += ocr_result[:2] + "-"
                    temp_text += ocr_result[2:4] + " "
                    if len(ocr_result[4:]) >= 5:
                        temp_text += ocr_result[4:7] + "." + ocr_result[7:]
                    else:
                        temp_text += ocr_result[4:]
            elif self.license_plate_format == 3:
                # bien quan doi 2 dong
                if ocr_result[:2].isalpha():
                    temp_text += ocr_result[:2] + "-" + ocr_result[2:]
        else:
            if self.license_plate_format == 1:
                # bien dan dung 1 dong
                if ocr_result[:2].isdigit() and ocr_result[2].isalpha():
                    temp_text += ocr_result[:3] + "-"
                    if len(ocr_result[3:]) >= 5:
                        temp_text += ocr_result[3:6] + "." + ocr_result[6:]
                    else:
                        temp_text += ocr_result[3:]
            elif self.license_plate_format == 3:
                # bien quan doi 1 dong
                if ocr_result[:2].isalpha():
                    temp_text += ocr_result[:2] + "-" + ocr_result[2:]
        if temp_text != "":
            return temp_text
        else:
            return ocr_result
