import cv2
import numpy as np
import torch
import torch.nn.functional as F

from frigate.plate_detectors.alpr.retina_plate.detect_config import cfg_plate
from frigate.plate_detectors.alpr.retina_plate.layers.functions.prior_box import (
    PriorBox,
)
from frigate.plate_detectors.alpr.retina_plate.models.retinaplate import RetinaPlate
from frigate.plate_detectors.alpr.retina_plate.utils.box_utils import (
    decode,
    decode_landm,
)
from frigate.plate_detectors.alpr.retina_plate.utils.nms.py_cpu_nms import py_cpu_nms


class Plate_Detector:
    def __init__(self, load_to_cpu=False):
        self.debug = False
        self.model = self.load_model(cfg_plate, load_to_cpu)
        self.config = cfg_plate
        self.image_size = (
            cfg_plate["max_size"],
            cfg_plate["max_size"]
            * (cfg_plate["image_ratio"][1] / cfg_plate["image_ratio"][0]),
        )  # width, height

        priorbox = PriorBox(
            self.config,
            image_size=(self.image_size[1], self.image_size[0]),
            phase="test",
        )  # height, width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        priors = priorbox.forward()
        if self.config["gpu_inference"]:
            priors = priors.to(self.device)
            self.model = self.model.to(self.device)
        self.prior_data = priors.data
        self.scale = torch.Tensor(
            [
                self.image_size[0],
                self.image_size[1],
                self.image_size[0],
                self.image_size[1],
            ]
        )
        self.origin_size = None

    def get_input(self, img, debug=False):
        """
        Pre-process an image to match with model's input.
        :param img: cv2 image. channel order is BGR
        :param debug: True or False. Save resized image to the model
        :return:
        """
        self.origin_size = img.shape[0:2]
        max_size = self.config["max_size"]
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(max_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if resize != 1:
            img = cv2.resize(
                img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
            )
        im_height, im_width, _ = img.shape
        self.image_size = (im_width, im_height)

        # Update Anchor Box for new custom size
        if (im_width / im_height) != (
            self.config["image_ratio"][1] / self.config["image_ratio"][0]
        ):
            # print("Updated anchor box")
            self.config["image_ratio"][1], self.config["image_ratio"][0] = (
                im_width,
                im_height,
            )
            priorbox = PriorBox(
                self.config,
                image_size=(self.image_size[1], self.image_size[0]),
                phase="test",
            )  # height, width
            priors = priorbox.forward()
            if self.config["gpu_inference"]:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                priors = priors.to(self.device)
                self.model = self.model.to(self.device)
            self.prior_data = priors.data

        if debug:
            cv2.imwrite("debug/resized_image.jpg", img)
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img)

    def load_model(self, config, load_to_cpu=False):
        def remove_prefix(state_dict, prefix):
            """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
            print("remove prefix '{}'".format(prefix))

            def f(x):
                return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

            return {f(key): value for key, value in state_dict.items()}

        def check_keys(model, pretrained_state_dict):
            ckpt_keys = set(pretrained_state_dict.keys())
            model_keys = set(model.state_dict().keys())
            used_pretrained_keys = model_keys & ckpt_keys
            unused_pretrained_keys = ckpt_keys - model_keys
            missing_keys = model_keys - ckpt_keys
            print("Missing keys:{}".format(len(missing_keys)))
            print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
            print("Used keys:{}".format(len(used_pretrained_keys)))
            assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
            return True

        print("Loading pretrained model from {}".format(config["path_pth"]))
        if load_to_cpu:
            pretrained_dict = torch.load(
                config["path_pth"], map_location=lambda storage, loc: storage
            )
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                config["path_pth"],
                map_location=lambda storage, loc: storage.cuda(device),
            )
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
        else:
            pretrained_dict = remove_prefix(pretrained_dict, "module.")
        net = RetinaPlate(cfg=config, phase="test")
        check_keys(net, pretrained_dict)
        net.load_state_dict(pretrained_dict, strict=False)
        net.eval()
        return net

    def detect(self, input_image):
        """
        :param input_image: the processed image from get_input function
        :return:
        """
        if self.config["gpu_inference"]:
            input_image = input_image.to(self.device)
        loc, conf, landms = self.model(input_image)
        return loc, conf, landms

    def post_process(self, detection_boxes, detection_scores, detection_landmark):
        """

        :return:
        """
        # detection_scores = torch.from_numpy(detection_scores)
        detection_scores = F.softmax(detection_scores, dim=-1)
        boxes = decode(
            detection_boxes.squeeze(0), self.prior_data, self.config["variance"]
        )
        boxes[:, 0::2] = boxes[:, 0::2] * self.origin_size[1]  # width
        boxes[:, 1::2] = boxes[:, 1::2] * self.origin_size[0]  # height

        landms = decode_landm(
            detection_landmark.squeeze(0), self.prior_data, self.config["variance"]
        )
        landms[:, 0::2] = landms[:, 0::2] * self.origin_size[1]
        landms[:, 1::2] = landms[:, 1::2] * self.origin_size[0]

        if not self.config["gpu_inference"]:
            detection_scores = detection_scores.squeeze(0).detach().numpy()
            boxes = boxes.detach().numpy()
            landms = landms.detach().numpy()
        else:
            detection_scores = detection_scores.squeeze(0).cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()
            landms = landms.cpu().detach().numpy()
        scores = detection_scores[:, 1]
        # ignore low scores
        inds = np.where(scores > self.config["confidence_threshold"])[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: self.config["top_k"]]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.config["nms_threshold"])
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[: self.config["keep_top_k"], :]
        landms = landms[: self.config["keep_top_k"], :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets


if __name__ == "__main__":
    detector = Plate_Detector()
    print("Done")


if __name__ == "__main__":
    detector = Plate_Detector()
    print("Done")
