import cv2
import time
from alpr import AutoLPR
from retina_plate.utils.utils import draw_result
from motion_checker import Motion_Checker

#
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    boss = AutoLPR(gpu_id=0)
    fps = 0
    # Load image
    stream = cv2.VideoCapture('/Users/cannguyen/Downloads/test_alpr_2.mp4')
    br_model = Motion_Checker()
    while True:
        start_time = time.time()
        r, image = stream.read()
        plate_name = []
        detect_time = time.time()
        height_ratio = 1.0
        width_ratio = 1.0
        image = cv2.resize(image, (850, 480))
        _, background_image = br_model.run(image)
        result = boss.predict(image, background_image, height_ratio=height_ratio, width_ratio=width_ratio)
        height, width, channels = image.shape
        x_new = int((1 - width_ratio) * (width / 2))
        y_new = int((1 - height_ratio) * (height / 2))
        cv2.line(image, (x_new, y_new), (width - x_new, y_new), (0, 255, 0), thickness=1)
        cv2.line(image, (x_new, y_new), (x_new, height - y_new), (0, 255, 0), thickness=1)
        cv2.line(image, (width - x_new, y_new), (width - x_new, height - y_new), (0, 255, 0), thickness=1)
        cv2.line(image, (x_new, height - y_new), (width - x_new, height - y_new), (0, 255, 0), thickness=1)
        # print(result)
        # image = boss.processFrame(image)
        if result is not None:
            # Process to add -,. to ocr result
            result[-1] = boss.post_process(result[-1])
            image = draw_result(image, result[:-1], result[-1], 1)

        # time.sleep(0.01)
        cv2.imshow('Test', image)
        # cv2.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imshow("plate", plate)
    # cv2.waitKey()
