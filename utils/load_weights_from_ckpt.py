import tensorflow as tf


def load_from_google_bert(model, init_checkpoint):
    """
    the checkpoint file should be the same with https://github.com/google-research/bert
    :param model: your bert task model
    :param init_checkpoint: ckpt path, like path/to/bert_model.ckpt
    """
    variables = tf.train.load_checkpoint(init_checkpoint)

    # embedding weights
    model._encoder_layer.get_layer("word_embeddings").set_weights([
        variables.get_tensor("bert/embeddings/word_embeddings")])
    model._encoder_layer.get_layer("position_embeddings").set_weights([
        variables.get_tensor("bert/embeddings/position_embeddings")])
    model._encoder_layer.get_layer("type_embeddings").set_weights([
        variables.get_tensor("bert/embeddings/token_type_embeddings")])

    model._encoder_layer.get_layer("embeddings/layer_norm").set_weights([
        variables.get_tensor("bert/embeddings/LayerNorm/beta"),
        variables.get_tensor("bert/embeddings/LayerNorm/gamma")
    ])

    # multi attention weights
    for i in range(model._config['bert_config'].num_hidden_layers):
        model._encoder_layer.get_layer("transformer/layer_{}".format(i)).set_weights([
            tf.reshape(variables.get_tensor(
                "bert/encoder/layer_{}/attention/self/query/kernel".format(i)),
                [model.bert_config.hidden_size, model.bert_config.num_attention_heads, -1]),
            tf.reshape(
                variables.get_tensor("bert/encoder/layer_{}/attention/self/query/bias".format(i)),
                [model.bert_config.num_attention_heads, -1]),
            tf.reshape(variables.get_tensor(
                "bert/encoder/layer_{}/attention/self/key/kernel".format(i)),
                [model.bert_config.hidden_size, model.bert_config.num_attention_heads, -1]),
            tf.reshape(
                variables.get_tensor("bert/encoder/layer_{}/attention/self/key/bias".format(i)),
                [model.bert_config.num_attention_heads, -1]),
            tf.reshape(variables.get_tensor(
                "bert/encoder/layer_{}/attention/self/value/kernel".format(i)),
                [model.bert_config.hidden_size, model.bert_config.num_attention_heads, -1]),
            tf.reshape(
                variables.get_tensor("bert/encoder/layer_{}/attention/self/value/bias".format(i)),
                [model.bert_config.num_attention_heads, -1]),
            tf.reshape(variables.get_tensor(
                "bert/encoder/layer_{}/attention/output/dense/kernel".format(i)),
                [model.bert_config.num_attention_heads, -1, model.bert_config.hidden_size]),
            variables.get_tensor("bert/encoder/layer_{}/attention/output/dense/bias".format(i)),
            variables.get_tensor("bert/encoder/layer_{}/attention/output/LayerNorm/beta".format(i)),
            variables.get_tensor("bert/encoder/layer_{}/attention/output/LayerNorm/gamma".format(i)),
            variables.get_tensor("bert/encoder/layer_{}/intermediate/dense/kernel".format(i)),
            variables.get_tensor("bert/encoder/layer_{}/intermediate/dense/bias".format(i)),
            variables.get_tensor("bert/encoder/layer_{}/output/dense/kernel".format(i)),
            variables.get_tensor("bert/encoder/layer_{}/output/dense/bias".format(i)),
            variables.get_tensor("bert/encoder/layer_{}/output/LayerNorm/beta".format(i)),
            variables.get_tensor("bert/encoder/layer_{}/output/LayerNorm/gamma".format(i)),
        ])

    model._encoder_layer.get_layer("pooler_transform").set_weights([
        variables.get_tensor("bert/pooler/dense/kernel"),
        variables.get_tensor("bert/pooler/dense/bias"),
    ])

    init_vars = tf.train.list_variables(init_checkpoint)
    for name, shape in init_vars:
        if name.startswith("bert"):
            print(f"{name}, shape={shape}, *INIT FROM CKPT SUCCESS*")


def load_from_google_albert(model, init_checkpoint):
    """
        the checkpoint file should be the same with https://github.com/google-research/albert
        :param model: your albert task model
        :param init_checkpoint: ckpt path, like path/to/bert_model.ckpt
        """
    variables = tf.train.load_checkpoint(init_checkpoint)
    # embedding weights
    model._encoder_layer.get_layer("word_embeddings").set_weights([
        variables.get_tensor("bert/embeddings/word_embeddings")])
    model._encoder_layer.get_layer("position_embeddings").set_weights([
        variables.get_tensor("bert/embeddings/position_embeddings")])
    model._encoder_layer.get_layer("type_embeddings").set_weights([
        variables.get_tensor("bert/embeddings/token_type_embeddings")])

    model._encoder_layer.get_layer("embeddings/layer_norm").set_weights([
        variables.get_tensor("bert/embeddings/LayerNorm/beta"),
        variables.get_tensor("bert/embeddings/LayerNorm/gamma")
    ])

    model._encoder_layer.get_layer("embedding_projection").set_weights([
        variables.get_tensor("bert/encoder/embedding_hidden_mapping_in/kernel"),
        variables.get_tensor("bert/encoder/embedding_hidden_mapping_in/bias")
    ])

    # multi attention weights
    for i in range(model._config['bert_config'].num_hidden_layers):
        model._encoder_layer.get_layer("transformer/layer_{}".format(i)).set_weights([
            tf.reshape(variables.get_tensor(
                "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel"),
                [model.bert_config.hidden_size, model.bert_config.num_attention_heads, -1]),
            tf.reshape(
                variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias"),
                [model.bert_config.num_attention_heads, -1]),
            tf.reshape(variables.get_tensor(
                "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel"),
                [model.bert_config.hidden_size, model.bert_config.num_attention_heads, -1]),
            tf.reshape(
                variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias"),
                [model.bert_config.num_attention_heads, -1]),
            tf.reshape(variables.get_tensor(
                "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel"),
                [model.bert_config.hidden_size, model.bert_config.num_attention_heads, -1]),
            tf.reshape(
                variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias"),
                [model.bert_config.num_attention_heads, -1]),
            tf.reshape(variables.get_tensor(
                "bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel"),
                [model.bert_config.num_attention_heads, -1, model.bert_config.hidden_size]),
            variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias"),
            variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta"),
            variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma"),
            variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel"),
            variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias"),
            variables.get_tensor(
                "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel"),
            variables.get_tensor(
                "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias"),
            variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta"),
            variables.get_tensor("bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma"),
        ])

    model._encoder_layer.get_layer("pooler_transform").set_weights([
        variables.get_tensor("bert/pooler/dense/kernel"),
        variables.get_tensor("bert/pooler/dense/bias"),
    ])

    init_vars = tf.train.list_variables(init_checkpoint)
    for name, shape in init_vars:
        if name.startswith("bert"):
            print(f"{name}, shape={shape}, *INIT FROM CKPT SUCCESS*")
