import tensorflow as tf
from utils import tf_utils
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

    # def build(self, input_shape=None):
    #
    #     super(BertClassifier, self).build(
    #         [[None, self._config["sequence_length"]], [None, self._config["sequence_length"]],
    #          [None, self._config["sequence_length"]]])

    def call(self, inputs):
        inputs = [inputs["input_word_ids"], inputs["input_mask"], inputs["input_type_ids"]]

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
        variables = tf.train.load_checkpoint(checkpoint_file)
        # embedding weights
        self._encoder_layer.get_layer("word_embeddings").set_weights([
            variables.get_tensor("bert/embeddings/word_embeddings")])
        self._encoder_layer.get_layer("position_embeddings").set_weights([
            variables.get_tensor("bert/embeddings/position_embeddings")])
        self._encoder_layer.get_layer("type_embeddings").set_weights([
            variables.get_tensor("bert/embeddings/token_type_embeddings")])

        self._encoder_layer.get_layer("embeddings/layer_norm").set_weights([
            variables.get_tensor("bert/embeddings/LayerNorm/beta"),
            variables.get_tensor("bert/embeddings/LayerNorm/gamma")
        ])

        self._encoder_layer.get_layer("embedding_projection").set_weights([
            variables.get_tensor("bert/encoder/embedding_hidden_mapping_in/kernel"),
            variables.get_tensor("bert/encoder/embedding_hidden_mapping_in/bias")
        ])

        # multi attention weights
        for i in range(self._config['bert_config'].num_hidden_layers):
            self._encoder_layer.get_layer("transformer/layer_{}".format(i)).set_weights([
                tf.reshape(variables.get_tensor(
                    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel"),
                    [self.bert_config.hidden_size, self.bert_config.num_attention_heads, -1]),
                tf.reshape(
                    variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias"),
                    [self.bert_config.num_attention_heads, -1]),
                tf.reshape(variables.get_tensor(
                    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel"),
                    [self.bert_config.hidden_size, self.bert_config.num_attention_heads, -1]),
                tf.reshape(
                    variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias"),
                    [self.bert_config.num_attention_heads, -1]),
                tf.reshape(variables.get_tensor(
                    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel"),
                    [self.bert_config.hidden_size, self.bert_config.num_attention_heads, -1]),
                tf.reshape(
                    variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias"),
                    [self.bert_config.num_attention_heads, -1]),
                tf.reshape(variables.get_tensor(
                    "bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel"),
                    [self.bert_config.num_attention_heads, -1, self.bert_config.hidden_size]),
                variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias"),
                # variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta"),
                # variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma"),
                variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel"),
                variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias"),
                variables.get_tensor(
                    "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel"),
                variables.get_tensor(
                    "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias"),
                # variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta"),
                # variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma"),
            ])

        # self._encoder_layer.get_layer("pooler_transform").set_weights([
        #     variables.get_tensor("bert/pooler/dense/kernel"),
        #     variables.get_tensor("bert/pooler/dense/bias"),
        # ])

        init_vars = tf.train.list_variables(checkpoint_file)
        for name, shape in init_vars:
            if name.startswith("bert"):
                print(f"{name}, shape={shape}, *INIT FROM CKPT SUCCESS*")

