import tensorflow as tf
from utils import tf_utils


class SelfAttentionMask(tf.keras.layers.Layer):
    """Create 3D attention mask from a 2D tensor mask.

      inputs[0]: from_tensor: 2D or 3D Tensor of shape
        [batch_size, from_seq_length, ...].
      inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].

      Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """

    def call(self, inputs):
        from_tensor, to_mask = inputs
        from_shape = tf_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]

        to_shape = tf_utils.get_shape_list(to_mask, expected_rank=2)
        to_seq_length = to_shape[1]

        to_mask = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
            dtype=from_tensor.dtype)

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        #
        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = tf.ones(
            shape=[batch_size, from_seq_length, 1], dtype=from_tensor.dtype)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask
