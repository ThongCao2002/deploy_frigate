import cv2
import numpy as np
import random
import time


class Motion_Checker:
    def __init__(self):
        """
            Fantastic motion detection and how to find them
        """
        self.background_model = cv2.createBackgroundSubtractorKNN(history=300, dist2Threshold=100)
        self.max_size = 300
        self.motion_flag = False

    def run(self, image):
        """
        This function to return if input image is background only or having action inside
        :param image: BGR image.
        :return: True if having motion. False if background only
        """
        # Cropping image
        # cv2.imshow("test 1", image)
        # height, width, channels = image.shape
        # print(image.shape)
        # x_new = int((1 - 0.85) * (width / 2))
        # y_new = int((1 - 0.85) * (height / 2))
        # image = image[y_new: height - y_new, x_new: width - x_new]
        im_shape = image.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(self.max_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        if resize != 1:
            image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = image.shape
        background_img = self.background_model.apply(image)
        _, background_img = cv2.threshold(background_img, 1, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        background_img = cv2.morphologyEx(background_img, cv2.MORPH_OPEN, kernel)
        white_count = cv2.countNonZero(background_img)
        if white_count > 1000:
            print("Have action!")
            self.motion_flag = True
        else:
            print("Background only!")
            self.motion_flag = False
        # # Below code to make nice image to display
        # background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)
        # background_img = cv2.bitwise_or(image, background_img)
        return self.motion_flag, background_img


if __name__ == "__main__":
    br_model = Motion_Checker()
    cap = cv2.VideoCapture("/home/can/AI_Camera/License_Plate/alpr/test_video/Sequence 01_2.mp4")
    # cap = cv2.VideoCapture(0)
    fps = 0
    while True:
        ret, frame = cap.read()
        start_time = time.time()
        background_image = br_model.run(frame)
        rand = random.randint(0, 10)

        if rand < 3:
            fps = int(1.0 / (time.time() - start_time))
        cv2.putText(background_image, str(fps), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        cv2.imshow('frame', background_image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
