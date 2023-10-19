import os
from frigate.plate_detectors.alpr.plate_recognizer import Plate_Recognizer
import tensorflow as tf

trained_checkpoint_prefix = 'frigate/plate_detectors/alpr/LPRnet/weight/weight_tensorflow/LPRnet_steps515000_loss_1.729.ckpt'
export_dir = 'frigate/plate_detectors/alpr/models/lprnet/1'

recognition = Plate_Recognizer()
# loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
# loader.restore(sess, trained_checkpoint_prefix)

# Export checkpoint to SavedModel
tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(recognition.model.inputs)
tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(recognition.model.dense_decoded)
prediction_signature = (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
    inputs={"inputs": tensor_info_x}, outputs={"outputs": tensor_info_y}, method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME))
builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
builder.add_meta_graph_and_variables(recognition.session,
                                        [tf.saved_model.SERVING],
                                        signature_def_map={"classification":prediction_signature},
                                        main_op=tf.compat.v1.tables_initializer(),
                                        strip_default_attrs=True)
builder.save()