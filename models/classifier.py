import tensorflow as tf
from utils import tf_utils, load_weights_from_ckpt
from configs import AlbertConfig, BertConfig
import layers


class BertClassifier(tf.keras.Model):

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

        self._predictions = tf.keras.layers.Activation(tf.nn.log_softmax)

        self._encoder_layer.build([[None, self._config["sequence_length"]], [None, self._config["sequence_length"]],
                                   [None, self._config["sequence_length"]]])
        self._logits_layer.build([None, self.bert_config.hidden_size])
        self._predictions.build([None, self._config['num_classes']])

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

    def _get_transformer_encoder(self, bert_config, sequence_length):
        """get transformer encoder model
        """
        kwargs = dict(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            num_hidden_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            activation=tf_utils.get_activation(bert_config.hidden_act),
            dropout_rate=bert_config.hidden_dropout_prob,
            attention_dropout_rate=bert_config.attention_probs_dropout_prob,
            sequence_length=sequence_length,
            max_sequence_length=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range),
            name="transformer_encoder")
        if isinstance(bert_config, AlbertConfig):
            kwargs['embedding_width'] = bert_config.embedding_size
            kwargs['num_hidden_groups'] = bert_config.num_hidden_groups
            return layers.AlbertTransformerEncoder(**kwargs)
        else:
            assert isinstance(bert_config, BertConfig)
            return layers.TransformerEncoder(**kwargs)

    def init_pre_training_weights(self, checkpoint_file):
        """init bert weights from pre training checkpoint
        """
        if isinstance(self.bert_config, AlbertConfig):
            load_weights_from_ckpt.load_from_google_albert(self, checkpoint_file)
        else:
            load_weights_from_ckpt.load_from_google_bert(self, checkpoint_file)
