import tensorflow as tf
from models import BaseModel


class BertClassifier(BaseModel):

    def __init__(self,
                 bert_config,
                 sequence_length,
                 num_classes,
                 initializer='glorot_uniform',
                 output='logits',
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.bert_config = bert_config
        self._config = {
            'bert_config': bert_config,
            'sequence_length': sequence_length,
            'num_classes': num_classes,
            'initializer': initializer,
            'output': output,
        }

        self._encoder_layer = self._get_transformer_encoder(bert_config, sequence_length)

        self._cls_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self._logits_layer = tf.keras.layers.Dense(
            num_classes,
            activation=None,
            kernel_initializer=initializer,
            name='predictions/transform/logits')

        self._encoder_layer.build([[None, self._config["sequence_length"]], [None, self._config["sequence_length"]],
                                   [None, self._config["sequence_length"]]])
        self._logits_layer.build([None, self.bert_config.hidden_size])

    def call(self, inputs):

        inputs = [tf.cast(inputs["input_word_ids"], tf.int32), tf.cast(inputs["input_mask"], tf.int32),
                  tf.cast(inputs["input_type_ids"], tf.int32)]

        _, cls_output = self._encoder_layer(inputs)

        cls_output = self._cls_dropout(cls_output)

        logits = self._logits_layer(cls_output)

        predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(logits)

        if self._config['output'] == 'logits':
            return logits
        elif self._config['output'] == 'predictions':
            return predictions

        raise ValueError(('Unknown `output` value "%s". `output` can be either "logits" or '
                          '"predictions"') % self._config['output'])
