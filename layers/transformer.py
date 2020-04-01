import tensorflow as tf

from layers import attention
from layers import dense_einsum


class Transformer(tf.keras.layers.Layer):
    """Transformer layer.

    This layer implements the Transformer from "Attention Is All You Need".
    (https://arxiv.org/abs/1706.03762).

    Arguments:
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate layer.
      intermediate_activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability for the post-attention and output dropout.
      attention_dropout_rate: Dropout probability for within the attention layer.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
    """

    def __init__(self,
                 num_attention_heads,
                 intermediate_size,
                 intermediate_activation,
                 dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self._num_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._intermediate_activation = intermediate_activation
        self._attention_dropout_rate = attention_dropout_rate
        self._dropout_rate = dropout_rate
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        input_tensor = input_shape[0] if len(input_shape) == 2 else input_shape
        input_tensor_shape = tf.TensorShape(input_tensor)
        if len(input_tensor_shape) != 3:
            raise ValueError("TransformerLayer expects a three-dimensional input of "
                             "shape [batch, sequence, width].")
        batch_size, sequence_length, hidden_size = input_tensor_shape

        if len(input_shape) == 2:
            mask_tensor_shape = tf.TensorShape(input_shape[1])
            expected_mask_tensor_shape = tf.TensorShape(
                [batch_size, sequence_length, sequence_length])
            if not expected_mask_tensor_shape.is_compatible_with(mask_tensor_shape):
                raise ValueError("When passing a mask tensor to TransformerLayer, the "
                                 "mask tensor must be of shape [batch, "
                                 "sequence_length, sequence_length] (here %s). Got a "
                                 "mask tensor of shape %s." %
                                 (expected_mask_tensor_shape, mask_tensor_shape))
        if hidden_size % self._num_heads != 0:
            raise ValueError(
                "The input size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, self._num_heads))
        self._attention_head_size = int(hidden_size // self._num_heads)

        self._attention_layer = attention.MultiHeadAttention(
            num_heads=self._num_heads,
            head_size=self._attention_head_size,
            dropout_rate=self._attention_dropout_rate,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="transformer/self_attention")
        self._attention_layer.build((input_shape[0], input_shape[0]))

        self._attention_output_dense = dense_einsum.DenseEinsum(
            output_shape=hidden_size,
            num_summed_dimensions=2,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="transformer/self_attention_output")
        self._attention_output_dense.build(input_shape[0][:2] + [12, input_shape[0][-1]//12])

        self._attention_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
        # Use float32 in layernorm for numeric stability.
        # It is probably safe in mixed_float16, but we haven't validated this yet.
        self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="transformer/self_attention_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32)
        self._intermediate_dense = dense_einsum.DenseEinsum(
            output_shape=self._intermediate_size,
            activation=None,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="transformer/intermediate")
        self._intermediate_dense.build(input_shape[0])

        # Use float32 in intermediate gelu activation for numeric stability.
        # TODO(b/149117297): investigate gelu numeric stability.
        self._intermediate_activation_layer = tf.keras.layers.Activation(
            self._intermediate_activation, dtype=tf.float32)
        self._output_dense = dense_einsum.DenseEinsum(
            output_shape=hidden_size,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="transformer/output")
        self._output_dense.build(input_shape[0][:-1]+[self._intermediate_size])

        self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
        # Use float32 in layernorm for numeric stability.
        self._output_layer_norm = tf.keras.layers.LayerNormalization(
            name="transformer/output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)

        super(Transformer, self).build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            input_tensor, attention_mask = inputs
        else:
            input_tensor, attention_mask = (inputs, None)

        attention_inputs = [input_tensor, input_tensor]

        if attention_mask is not None:
            attention_inputs.append(attention_mask)
        attention_output = self._attention_layer(attention_inputs)
        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=True)
        attention_output = self._attention_layer_norm(input_tensor +
                                                      attention_output)
        intermediate_output = self._intermediate_dense(attention_output)
        intermediate_output = self._intermediate_activation_layer(intermediate_output)
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output, training=True)
        # During mixed precision training, attention_output is from layer norm and
        # is always fp32 for now. Cast layer_output to fp32 for the subsequent
        # add.
        layer_output = tf.cast(layer_output, tf.float32)
        layer_output = self._output_layer_norm(layer_output + attention_output)

        return layer_output

    def get_config(self):
        config = {
            "num_attention_heads":
                self._num_heads,
            "intermediate_size":
                self._intermediate_size,
            "intermediate_activation":
                self._intermediate_activation,
            "dropout_rate":
                self._dropout_rate,
            "attention_dropout_rate":
                self._attention_dropout_rate,
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                tf.keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer":
                tf.keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint":
                tf.keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint":
                tf.keras.constraints.serialize(self._bias_constraint)
        }
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))