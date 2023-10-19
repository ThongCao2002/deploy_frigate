import onnx
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from retina_plate.detect_config import cfg_plate
from caffe2.python.onnx import backend
from retina_plate.layers.functions.prior_box import PriorBox
from retina_plate.utils.box_utils import decode, decode_landm
from retina_plate.utils.nms.py_cpu_nms import py_cpu_nms


class Plate_Detector:
    def __init__(self):
        self.debug = False
        self.model = self.load_model(cfg_plate['path'])
        self.config = cfg_plate
        self.image_size = self.config['image_size']  # width, height
        self.scale = torch.Tensor([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]])
        self.origin_size = None

    def get_input(self, img_raw, debug=False):
        """
        Pre-process an image to match with model's input.
        :param img_raw: cv2 image. channel order is BGR
        :param debug: True or False. Save resized image to the model
        :return:
        """
        self.origin_size = img_raw.shape[0:2]
        img = cv2.resize(img_raw, self.image_size)  # img: 480, 850, 3
        # img = img_raw
        if debug:
            cv2.imwrite('debug/resized_image.jpg', img)
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def load_model(self, path_to_model):
        """
        Read Onnx Retina model, prepare a caffe2 backend to run model in PC
        :param path_to_model:
        :return:
        """
        predictor = onnx.load(path_to_model)
        onnx.checker.check_model(predictor)
        if self.debug:
            print(onnx.helper.printable_graph(predictor.graph))
        predictor = backend.prepare(predictor, device="CPU")
        return predictor

    def detect(self, input_image):
        """
        :param input_image: the processed image from get_input function
        :return:
        """
        loc, conf, landms = self.model.run(input_image)
        return loc, conf, landms

    def post_process(self, detection_boxes, detection_scores, detection_landmark):
        """

        :return:
        """
        detection_scores = torch.from_numpy(detection_scores)
        detection_scores = F.softmax(detection_scores, dim=-1)
        detection_scores = detection_scores.numpy()
        priorbox = PriorBox(self.config, image_size=(self.image_size[1], self.image_size[0]), phase='test')  # height, width
        priors = priorbox.forward()
        # priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(torch.from_numpy(detection_boxes).data.squeeze(0), prior_data, self.config['variance'])
        boxes[:, 0::2] = boxes[:, 0::2] * self.origin_size[1]  # width
        boxes[:, 1::2] = boxes[:, 1::2] * self.origin_size[0]  # height
        boxes = boxes.cpu().numpy()
        scores = torch.from_numpy(detection_scores).data.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(torch.from_numpy(detection_landmark).data.squeeze(0), prior_data, self.config['variance'])
        landms[:, 0::2] = landms[:, 0::2] * self.origin_size[1]
        landms[:, 1::2] = landms[:, 1::2] * self.origin_size[0]
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.config['confidence_threshold'])[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.config['top_k']]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.config['nms_threshold'])
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.config['keep_top_k'], :]
        landms = landms[:self.config['keep_top_k'], :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets
