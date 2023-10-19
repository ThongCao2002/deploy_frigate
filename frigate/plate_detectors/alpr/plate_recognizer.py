import os

import cv2
import numpy as np

from .LPRnet.LPRnet import *


class Plate_Recognizer:
    def __init__(self):
        """ """
        self.model, self.session = self.load_model(
            os.path.dirname(os.path.abspath(__file__))
            + "/LPRnet/weight/weight_tensorflow/LPRnet_steps515000_loss_1.729.ckpt"
        )
        self.input_size = (94, 24)

    def get_input(self, img, flip=False):
        """
        Pre-process the input plate
        :param flip:
        :param img: BGR opencv image
        :return:
        """

        # TODO: cut image into text lines.

        img = cv2.resize(img, self.input_size)
        img = np.expand_dims(img, axis=0)
        return img

    def load_model(self, path_to_model):
        """
        Load face recognition model
        :param path_to_model:
        :return:
        """
        lprnet = LPRnet(is_train=False)

        def restore_checkpoint(sess, saver, ckpt, is_train=True):
            try:
                saver.restore(sess, ckpt)
                print("restore from checkpoint: {}".format(ckpt))
                return True
            except:
                if is_train:
                    print("train from scratch")
                else:
                    print("no valid checkpoint provided")
                return False

        # import os

        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(lprnet.init)
        saver = tf.train.Saver(tf.global_variables())

        if not restore_checkpoint(sess, saver, path_to_model, is_train=False):
            return
        return lprnet, sess

    def run(self, input_image):
        """

        :return:
        """
        # with self.session as sess:
        test_feed = {self.model.inputs: input_image}
        dense_decode = self.session.run(self.model.dense_decoded, test_feed)

        return dense_decode

    @staticmethod
    def post_process(model_output):
        decoded_labels = []
        for item in model_output:
            expression = ["" if i == -1 else DECODE_DICT[i] for i in item]
            expression = "".join(expression)
            decoded_labels.append(expression)

        return decoded_labels


if __name__ == "__main__":
    recognizer = Plate_Recognizer()
    print("test")
    print("test")
    print("test")
