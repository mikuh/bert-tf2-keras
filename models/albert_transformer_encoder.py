import tensorflow as tf
from utils import activations
import layers


class AlbertTransformerEncoder(tf.keras.Model):
    """ALBERT (https://arxiv.org/abs/1810.04805) text encoder network.

      This network implements the encoder described in the paper "ALBERT: A Lite
      BERT for Self-supervised Learning of Language Representations"
      (https://arxiv.org/abs/1909.11942).

      Compared with BERT (https://arxiv.org/abs/1810.04805), ALBERT refactorizes
      embedding parameters into two smaller matrices and shares parameters
      across layers.

      The default values for this object are taken from the ALBERT-Base
      implementation described in the paper.

      Arguments:
        vocab_size: The size of the token vocabulary.
        embedding_width: The width of the word embeddings. If the embedding width
          is not equal to hidden size, embedding parameters will be factorized into
          two matrices in the shape of ['vocab_size', 'embedding_width'] and
          ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
          smaller than 'hidden_size').
        hidden_size: The size of the transformer hidden layers.
        num_layers: The number of transformer layers.
        num_attention_heads: The number of attention heads for each transformer. The
          hidden size must be divisible by the number of attention heads.
        sequence_length: The sequence length that this encoder expects. If None, the
          sequence length is dynamic; if an integer, the encoder will require
          sequences padded to this length.
        max_sequence_length: The maximum sequence length that this encoder can
          consume. If None, max_sequence_length uses the value from sequence length.
          This determines the variable shape for positional embeddings.
        type_vocab_size: The number of types that the 'type_ids' input can take.
        intermediate_size: The intermediate size for the transformer layers.
        activation: The activation to use for the transformer layers.
        dropout_rate: The dropout rate to use for the transformer layers.
        attention_dropout_rate: The dropout rate to use for the attention layers
          within the transformer layers.
        initializer: The initialzer to use for all weights in this encoder.
      """

    def __init__(self,
                 vocab_size,
                 embedding_width=128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 num_hidden_groups=1,
                 inner_group_num=1,
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
        super(AlbertTransformerEncoder, self).__init__(**kwargs)

        if inner_group_num != 1:
            raise ValueError("We only support 'inner_group_num' as 1.")

        if not max_sequence_length:
            max_sequence_length = sequence_length

        self._activation = activation
        self._config_dict = {
            'vocab_size': vocab_size,
            'embedding_width': embedding_width,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'num_hidden_groups': num_hidden_groups,
            'inner_group_num': inner_group_num,
            'sequence_length': sequence_length,
            'max_sequence_length': max_sequence_length,
            'type_vocab_size': type_vocab_size,
            'intermediate_size': intermediate_size,
            'activation': tf.keras.activations.serialize(activation),
            'dropout_rate': dropout_rate,
            'attention_dropout_rate': attention_dropout_rate,
            'initializer': tf.keras.initializers.serialize(initializer),
            'return_all_encoder_outputs': return_all_encoder_outputs
        }

        self._cls_output_layer = tf.keras.layers.Dense(
            units=hidden_size,
            activation='tanh',
            kernel_initializer=initializer,
            name='pooler_transform')


    def build(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()
        self._embedding_layer = layers.OnDeviceEmbedding(
            vocab_size=self._config_dict['vocab_size'],
            embedding_width=self._config_dict['embedding_width'],
            initializer=self._config_dict['initializer'],
            name='word_embeddings')

        # Always uses dynamic slicing for simplicity.
        self._position_embedding_layer = layers.PositionEmbedding(
            initializer=self._config_dict['initializer'],
            use_dynamic_slicing=True,
            max_sequence_length=self._config_dict['max_sequence_length'],
            name='position_embeddings')
        self._position_embedding_layer.build(input_shape[0] + [self._config_dict["embedding_width"]])

        self._type_embedding_layer = layers.OnDeviceEmbedding(
            vocab_size=self._config_dict['type_vocab_size'],
            embedding_width=self._config_dict['embedding_width'],
            initializer=self._config_dict['initializer'],
            use_one_hot=True,
            name='type_embeddings')

        self._embedding_layer_normalization = tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm',
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32)

        self._embedding_dropout = tf.keras.layers.Dropout(rate=self._config_dict['dropout_rate'])

        self._embedding_projection_layer = layers.DenseEinsum(
            output_shape=self._config_dict['hidden_size'],
            kernel_initializer=self._config_dict['initializer'],
            name='embedding_projection')
        self._embedding_projection_layer.build(input_shape=input_shape[0] + [self._config_dict['embedding_width']])

        self._self_attention_mask = layers.SelfAttentionMask()

        # transformer layer
        self._transformer_layers = []
        last_name = None
        for layer_idx in range(self._config_dict['num_hidden_layers']):
            group_idx = int(layer_idx / self._config_dict['num_hidden_layers'] * self._config_dict['num_hidden_groups'])
            if group_idx == last_name:
                layer = self._transformer_layers[-1]
            else:
                layer = layers.Transformer(
                    num_attention_heads=self._config_dict['num_attention_heads'],
                    intermediate_size=self._config_dict['intermediate_size'],
                    intermediate_activation=self._activation,
                    dropout_rate=self._config_dict['dropout_rate'],
                    attention_dropout_rate=self._config_dict['attention_dropout_rate'],
                    kernel_initializer=self._config_dict['initializer'],
                    name='transformer/layer_%d' % group_idx)
                layer.build([input_shape[0] + [self._config_dict['hidden_size']], input_shape[0] + input_shape[0][-1:]])
            last_name = group_idx
            self._transformer_layers.append(layer)

        self._cls_output_layer = tf.keras.layers.Dense(
            units=self._config_dict['hidden_size'],
            activation='tanh',
            kernel_initializer=self._config_dict['initializer'],
            name='pooler_transform')

        super().build(input_shape)

    def call(self, inputs):
        word_ids, mask, type_ids = inputs

        word_embeddings = self._embedding_layer(tf.cast(word_ids, tf.int32))
        position_embeddings = self._position_embedding_layer(tf.cast(word_embeddings, tf.int32))
        type_embeddings = self._type_embedding_layer(tf.cast(type_ids, tf.int32))

        embeddings = tf.keras.layers.Add()([word_embeddings, position_embeddings, type_embeddings])
        embeddings = self._embedding_layer_normalization(embeddings)
        embeddings = self._embedding_dropout(embeddings, training=True)

        # We project the 'embedding' output to 'hidden_size' if it is not already 'hidden_size'.
        if self._config_dict['embedding_width'] != self._config_dict['hidden_size']:
            embeddings = self._embedding_projection_layer(embeddings)

        data = embeddings
        attention_mask = self._self_attention_mask([data, mask])

        encoder_outputs = []
        for layer in self._transformer_layers:
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        first_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(encoder_outputs[-1])
        cls_output = self._cls_output_layer(first_token_tensor)

        if self._config_dict['return_all_encoder_outputs']:
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


if __name__ == '__main__':
    import numpy as np
    model = AlbertTransformerEncoder(vocab_size=20128, type_vocab_size=2, name="transformer_encoder")
    model.build([[None, 50], [None, 50], [None, 50]])
    # model([tf.ones([1, 50]), tf.ones([1, 50]), tf.ones([1, 50])])
    # input1 = tf.keras.Input(shape=(50,), dtype=tf.int32)
    # input2 = tf.keras.Input(shape=(50,), dtype=tf.int32)
    # input3 = tf.keras.Input(shape=(50,), dtype=tf.int32)
    #
    # outputs = encoder([input1, input2, input3])
    # model = tf.keras.Model(inputs=[input1, input2, input3], outputs=outputs)
    model.summary()


    for layer in model.layers:
        for weight in layer.get_weights():
            print(weight.shape)

    # print(model.get_layer(name="transformer/layer_0").get_weights())
    # print(encoder.get_weights())
    # for layer_weights in encoder.get_weights():
    #     print(layer_weights.shape)
