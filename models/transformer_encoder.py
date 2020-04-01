import tensorflow as tf
from utils import activations
import layers


class TransformerEncoder(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 sequence_length=512,
                 max_sequence_length=None,
                 type_vocab_size=16,
                 intermediate_size=3072,
                 activation=activations.gelu,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 return_all_encoder_outputs=False,
                 **kwargs):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer = initializer = tf.keras.initializers.get(initializer)

        self.return_all_encoder_outputs = return_all_encoder_outputs

        if not max_sequence_length:
            max_sequence_length = sequence_length

        # Always uses dynamic slicing for simplicity.
        self._position_embedding_layer = layers.PositionEmbedding(
            initializer=initializer,
            use_dynamic_slicing=True,
            max_sequence_length=max_sequence_length)

        self._type_embedding_layer = layers.OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=hidden_size,
            initializer=initializer,
            use_one_hot=True,
            name='type_embeddings')

        self._layer_normalization = tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm',
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32)

        self._dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self._self_attention_mask = layers.SelfAttentionMask()

        self._cls_dense = tf.keras.layers.Dense(
            units=hidden_size,
            activation='tanh',
            kernel_initializer=initializer,
            name='pooler_transform')

        self._transformer_layers = []
        for i in range(self.num_layers):
            layer = layers.Transformer(
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                intermediate_activation=self.activation,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                kernel_initializer=self.initializer,
                name='transformer/layer_%d' % i)
            self._transformer_layers.append(layer)

    def build(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        self._embedding_layer = layers.OnDeviceEmbedding(
            vocab_size=self.vocab_size,
            embedding_width=self.hidden_size,
            initializer=self.initializer,
            name='word_embeddings')

    def call(self, inputs):
        word_ids, mask, type_ids = inputs

        word_embeddings = self._embedding_layer(tf.cast(word_ids, tf.int32))
        position_embeddings = self._position_embedding_layer(tf.cast(word_embeddings, tf.int32))
        type_embeddings = self._type_embedding_layer(tf.cast(type_ids, tf.int32))

        embeddings = tf.keras.layers.Add()([word_embeddings, position_embeddings, type_embeddings])
        embeddings = self._layer_normalization(embeddings)
        embeddings = self._dropout(embeddings)

        data = embeddings
        attention_mask = self._self_attention_mask([data, mask])
        encoder_outputs = []
        for layer in self._transformer_layers:
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        first_token_tensor = (
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                encoder_outputs[-1]))
        cls_output = self._cls_dense(first_token_tensor)

        if self.return_all_encoder_outputs:
            outputs = [encoder_outputs, cls_output]
        else:
            outputs = [encoder_outputs[-1], cls_output]

        return outputs

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @classmethod
    def from_config(cls, config):
        return cls(**config)
