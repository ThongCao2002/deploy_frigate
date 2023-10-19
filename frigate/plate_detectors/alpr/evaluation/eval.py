import os
import glob
import numpy as np
import cv2
import argparse
import json
import editdistance
from plate_recognizer import Plate_Recognizer
from plate_detector import Plate_Detector
from plate_checker import Plate_Checker
from retina_plate.utils.utils import img_transform


def intersect(a, b):
    '''Determine the intersection of two rectangles'''
    rect = (0, 0, 0, 0)
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    # Do we have a valid intersection?
    if r1 > r0 and c1 > c0:
        rect = (r0, c0, r1, c1)
    return rect


def union(a, b):
    r0 = min(a[0], b[0])
    c0 = min(a[1], b[1])
    r1 = max(a[2], b[2])
    c1 = max(a[3], b[3])
    return (r0, c0, r1, c1)


def area(a):
    '''Computes rectangle area'''
    width = a[2] - a[0]
    height = a[3] - a[1]
    return abs(width * height)


def draw_detections(img, box, text, color=(255, 0, 0)):
    # draw2 = np.copy(img)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
    cx = box[0]
    cy = box[1] - 20
    cv2.putText(img, text, (cx, cy),
                cv2.FONT_HERSHEY_TRIPLEX, 1, color)
    return img


def load_gt(path, expand=8):
    label = []
    with open(path) as json_file:
        data = json.load(json_file)
        for p in data['objects']:
            x_min = p['bbox']['xmin']
            y_min = p['bbox']['ymin']
            x_max = p['bbox']['xmax']
            y_max = p['bbox']['ymax']
            w = p['bbox']['xmax'] - p['bbox']['xmin']
            h = p['bbox']['ymax'] - p['bbox']['ymin']
            text = p['lp'].strip().replace("/", "").upper()
            label.append(
                [int(x_min - w / expand), int(y_min - h / expand), int(x_max + w / expand), int(y_max + h / expand),
                 text])
        return label


def evaluate_image(image, detections, ground_truth, iou_th=0.5, iou_th_vis=0.5, iou_th_eval=0.5):
    '''
    Summary : Returns end-to-end true-positives, detection true-positives, number of GT to be considered for eval (len > 2).
    Description : For each predicted bounding-box, comparision is made with each GT entry. Values of number of end-to-end true
                                positives, number of detection true positives, number of GT entries to be considered for evaluation are computed.

    Parameters
    ----------
    image: opencv image
    detections : list of lists
            Tuple of predicted bounding boxes along with transcriptions and text/no-text score.
    ground_truth: list of lists
            ground truth which contains boxes and texts
    iou_th_eval : float
            Threshold value of intersection-over-union used for evaluation of predicted bounding-boxes
    iou_th_vis : float
            Threshold value of intersection-over-union used for visualization when transciption is true but IoU is lesser.
    iou_th : float
            Threshold value of intersection-over-union between GT and prediction.



    Returns
    -------
    tp : int
            Number of predicted bounding-boxes having IoU with GT greater than iou_th_eval.
    tp_e2e : int
            Number of predicted bounding-boxes having same transciption as GT and len > 2.
    gt_e2e : int
            Number of GT entries for which transcription len > 2.
    '''

    gt_to_detection = {}
    detection_to_gt = {}
    tp = 0
    tp_e2e = 0
    tp_e2e_ed1 = 0
    gt_e2e = 0

    gt_matches = np.zeros(len(ground_truth))
    gt_matches_ed1 = np.zeros(len(ground_truth))

    for i in range(0, len(detections)):
        det = detections[i]
        bbox = [det[0], det[1], det[2], det[3]]

        det_text = det[4]  # Predicted transcription for bounding-box

        for gt_no in range(len(ground_truth)):
            gt_box = ground_truth[gt_no]
            rect_gt = [gt_box[0], gt_box[1], gt_box[2], gt_box[3]]
            txt = gt_box[4]
            cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 3)  # GREEN: GT
            cx = gt_box[0]
            cy = gt_box[3] - 20
            cv2.putText(image, txt, (cx, cy),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
            # cv2.imwrite(os.path.join('evaluation/visualize', os.path.basename(image_path)), image)
            # cv2.imshow("test1", cv2.resize(image, (850, 480)))
            # cv2.waitKey()
            inter = intersect(bbox, rect_gt)  # Intersection of predicted and GT bounding-boxes
            uni = union(bbox, rect_gt)  # Union of predicted and GT bounding-boxes
            ratio = area(inter) / float(area(uni))  # IoU measure between predicted and GT bounding-boxes

            # 1). Visualize the predicted-bounding box if IoU with GT is higher than IoU threshold (iou_th) (Always required)
            # 2). Visualize the predicted-bounding box if transcription matches the GT and condition 1. holds
            # 3). Visualize the predicted-bounding box if transcription matches and IoU with GT is less than iou_th_vis and 1. and 2. hold
            if ratio > iou_th:
                if not gt_no in gt_to_detection:
                    gt_to_detection[gt_no] = [0, 0]

                edit_dist = editdistance.eval(det_text.upper(), txt.upper())
                if edit_dist <= 1:
                    gt_matches_ed1[gt_no] = 1
                    if edit_dist > 0:
                        draw_detections(image, bbox, "Nearly", color=(255, 0, 0))  # BLUE: nearly good
                        # cv2.imshow("test", cv2.resize(image, (850, 480)))
                        # cv2.waitKey()
                if edit_dist == 0:  # det_text.lower().find(txt.lower()) != -1:
                    gt_matches[gt_no] = 1  # Change this parameter to 1 when predicted transcription is correct.
                    draw_detections(image, bbox, "Good", color=(255, 255, 255))  # WHITE: good works
                    # cv2.imwrite(os.path.join('evaluation/visualize', os.path.basename(image_path)), image)
                    # cv2.imshow("test1", cv2.resize(image, (850, 480)))
                    # cv2.waitKey()
                tupl = gt_to_detection[gt_no]
                if tupl[0] < ratio:
                    tupl[0] = ratio
                    tupl[1] = i
                    detection_to_gt[i] = [gt_no, ratio, edit_dist]

    # Count the number of end-to-end and detection true-positives
    for gt_no in range(gt_matches.shape[0]):
        gt = gt_matches[gt_no]
        gt_ed1 = gt_matches_ed1[gt_no]
        gt_box = ground_truth[gt_no]
        txt = gt_box[4]
        gt_e2e += 1
        if gt == 1:
            tp_e2e += 1
        if gt_ed1 == 1:
            tp_e2e_ed1 += 1

        if gt_no in gt_to_detection:
            tupl = gt_to_detection[gt_no]
            if tupl[0] > iou_th_eval:  # Increment detection true-positive, if IoU is greater than iou_th_eval
                tp += 1
    return tp, tp_e2e, gt_e2e, tp_e2e_ed1, detection_to_gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir',
                        default='/home/can/AI_Camera/License_Plate/LP_Detection/data/test/outdoor_test/images')
    parser.add_argument('-out_dir', default='eval')
    parser.add_argument('-eval_text_length', type=int, default=3)

    args = parser.parse_args()
    detector = Plate_Detector()
    recognizer = Plate_Recognizer()
    checker = Plate_Checker()
    images = glob.glob(os.path.join(args.images_dir, '*.jpg'))
    png = glob.glob(os.path.join(args.images_dir, '*.png'))
    images.extend(png)
    png = glob.glob(os.path.join(args.images_dir, '*.JPG'))
    images.extend(png)

    tp_all = 0
    gt_all = 0
    tp_e2e_all = 0
    gt_e2e_all = 0
    tp_e2e_ed1_all = 0
    detecitons_all = 0

    im_no = 0
    min_height = 8

    eval_text_length = args.eval_text_length
    label_root = '/home/can/AI_Camera/License_Plate/LP_Detection/data/test/outdoor_test/jsons'
    nums = []
    image_no = 0
    for image_path in images:
        # image_path = '/home/can/AI_Camera/License_Plate/LP_Detection/data/test/outdoor_test/images/IMG_20200924_154616_t.jpg'
        print(image_path)
        image = cv2.imread(image_path)
        # get label
        label_file = os.path.join(label_root, os.path.basename(image_path)[:-4] + ".json")
        ground_truth = load_gt(label_file)
        for temp in ground_truth:
            cv2.rectangle(image, (temp[0], temp[1]), (temp[2], temp[3]), (0, 255, 0), 3)  # GREEN: GT
            cx = temp[0]
            cy = temp[3] - 20
            cv2.putText(image, temp[4], (cx, cy),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        # cv2.imshow("test", image)
        # cv2.waitKey()
        image_processed = detector.get_input(image)
        model_output = detector.detect(image_processed)
        detection_result = detector.post_process(model_output[0], model_output[1], model_output[2])
        detetcions_out = []
        plate_name = []
        if len(detection_result) != 0:
            for i in range(0, len(detection_result)):
                plate = img_transform(image, detection_result[i][5:])
                # cv2.imshow("plate", plate)
                # cv2.waitKey()
                # Do dilate to cut image into 2 lines (need to know if image is one or two lines)
                ocr_image_processed = recognizer.get_input(plate)
                ocr_model_out = recognizer.run(ocr_image_processed)
                ocr_result = recognizer.post_process(ocr_model_out)
                plate_name.append(ocr_result[0])
            for i, b in enumerate(detection_result):
                text = "{:.4f}".format(b[4])
                if not checker.run(plate_name[i]):
                    b = list(map(int, b))
                    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (150, 50, 150), 2)  # RED: detection
                    cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 1)  # red, top left
                    cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 1)  # yellow, top right
                    cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 1)  # purple, bottom left
                    cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 1)  # green, bottom right
                    cx = b[0]
                    cy = int((b[1] + b[3]) / 2)
                    cv2.putText(image, "REMOVE: " + plate_name[i], (cx, cy),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
                    continue
                b = list(map(int, b))
                cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # RED: detection
                cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 1)  # red, top left
                cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 1)  # yellow, top right
                cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 1)  # purple, bottom left
                cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 1)  # green, bottom right
                cx = b[0]
                cy = int((b[1] + b[3]) / 2)
                cv2.putText(image, plate_name[i], (cx, cy),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
                # cv2.imwrite(os.path.join('evaluation/visualize', os.path.basename(image_path)), image)
                detetcions_out.append([b[0], b[1], b[2], b[3], plate_name[i]])
        tp, tp_e2e, gt_e2e, tp_e2e_ed1, detection_to_gt = evaluate_image(image, detetcions_out, ground_truth)
        cv2.imwrite(os.path.join('evaluation/visualize', os.path.basename(image_path)), image)
        tp_all += tp
        gt_all += len(ground_truth)
        tp_e2e_all += tp_e2e
        gt_e2e_all += gt_e2e
        tp_e2e_ed1_all += tp_e2e_ed1
        detecitons_all += len(detetcions_out)

    print("	E2E recall {0:.3f} / precision: {1:.3f} \n Detect recall {2:.3f}, precision: {3:.3f}".format(
        tp_e2e_all / float(max(1, gt_e2e_all)),
        tp_e2e_all / float(max(1, detecitons_all)),
        tp_all / float(max(1, gt_all)),
        # tp_e2e_ed1_all / float(max(1, gt_e2e_all)),
        tp_all / float(max(1, detecitons_all))))
