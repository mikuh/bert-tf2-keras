import tensorflow as tf
from utils import tf_utils


class OnDeviceEmbedding(tf.keras.layers.Layer):
    """Performs an embedding lookup suitable for accelerator devices.

    This layer uses either tf.gather or tf.one_hot to translate integer indices to
    float embeddings.

    Arguments:
      vocab_size: Number of elements in the vocabulary.
      embedding_width: Output size of the embedding layer.
      initializer: The initializer to use for the embedding weights. Defaults to
        "glorot_uniform".
      use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
        lookup. Defaults to False (that is, using tf.gather). Setting this option
        to True may improve performance, especially on small vocabulary sizes,
        but will generally require more memory.
    """

    def __init__(self,
                 vocab_size,
                 embedding_width,
                 initializer="glorot_uniform",
                 use_one_hot=False,
                 **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super(OnDeviceEmbedding, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding_width = embedding_width
        self._initializer = initializer
        self._use_one_hot = use_one_hot

        self.embeddings = self.add_weight("embeddings",
                                          shape=[self._vocab_size, self._embedding_width],
                                          initializer=self._initializer)

    def call(self, inputs):
        input_shape = tf_utils.get_shape_list(inputs, expected_rank=2)  # (batch_size, sequence_length)
        input_shape.append(self._embedding_width)
        flat_inputs = tf.reshape(inputs, [-1])
        if self._use_one_hot:
            one_hot_data = tf.one_hot(
                flat_inputs, depth=self._vocab_size, dtype=self._dtype)
            embeddings = tf.matmul(one_hot_data, self.embeddings)
        else:
            embeddings = tf.gather(self.embeddings, flat_inputs)
        embeddings = tf.reshape(embeddings, input_shape)  # (batch_size, sequence_length, _embedding_width)

        return embeddings

    def get_config(self):
        config = {
            "vocab_size": self._vocab_size,
            "embedding_width": self._embedding_width,
            "initializer": self._initializer,
            "use_one_hot": self._use_one_hot,
        }
        base_config = super(OnDeviceEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    emb = OnDeviceEmbedding(10000, 128)

    import numpy as np

    print(emb.get_weights())

    emb.set_weights([np.ones((10000, 128))])
    print(emb.get_weights())

    print(emb.get_config())
