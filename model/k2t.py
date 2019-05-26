import keras as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

from keras.models import Model, load_model
from keras.utils.generic_utils import CustomObjectScope

def my_crossentropy(y_true, y_pred):
    return K.backend.mean(2*K.backend.abs(y_true-0.5) * K.backend.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.backend.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.backend.mean(mymask(y_true) * K.backend.square(K.backend.sqrt(y_pred) - K.backend.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.backend.mean(mymask(y_true) * (10*K.backend.square(K.backend.square(K.backend.sqrt(y_pred) - K.backend.sqrt(y_true))) + K.backend.square(K.backend.sqrt(y_pred) - K.backend.sqrt(y_true)) + 0.01*K.backend.binary_crossentropy(y_pred, y_true)), axis=-1)


with CustomObjectScope({'my_crossentropy': my_crossentropy,
                        'msse': msse, 'mycost': mycost}):
  sess = K.backend.get_session()
  model=load_model("rnn_noise_new.h5")

  constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)

  output="graph.pb"
  graph_io.write_graph(constant_graph, "/", output, as_text=False)
  print('saved the freezed graph (ready for inference) at: ', output)
