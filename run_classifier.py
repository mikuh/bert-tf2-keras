import tensorflow as tf
from utils import performance
from utils import optimization
from utils import keras_utils
from utils.data_utils import create_classifier_dataset
from utils import distribution_utils
from models import BertClassifier
from configs import AlbertConfig, BertConfig
import math
import os
import time
import keras.backend as K


def get_optimizer(initial_lr, steps_per_epoch, epochs, warmup_steps, use_float16=False):
    optimizer = optimization.create_optimizer(initial_lr, steps_per_epoch * epochs, warmup_steps)
    optimizer = performance.configure_optimizer(
        optimizer,
        use_float16=use_float16,
        use_graph_rewrite=False)
    return optimizer


def get_loss_fn(num_classes):
    """Gets the classification loss function."""

    def classification_loss_fn(labels, logits):
        """Classification loss."""
        # K.print_tensor(labels, message=',y_true = ')
        # K.print_tensor(logits, message=',y_predict = ')
        labels = tf.squeeze(labels)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(
            tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(
            tf.cast(one_hot_labels, dtype=tf.float32) * tf.cast(log_probs, tf.float32), axis=-1)
        return tf.reduce_mean(per_example_loss)

    return classification_loss_fn


def metric_fn():
    return tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy', dtype=tf.float32)


def get_callbacks(train_batch_size, log_steps, model_dir):
    custom_callback = keras_utils.TimeHistory(
        batch_size=train_batch_size,
        log_steps=log_steps,
        logdir=os.path.join(model_dir, 'logs'))

    summary_callback = tf.keras.callbacks.TensorBoard(os.path.join(model_dir, 'graph'), update_freq='batch')

    checkpoint_path = os.path.join(model_dir, 'checkpoint-{epoch:02d}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, save_freq=32000)

    return [custom_callback, summary_callback, checkpoint_callback]


if __name__ == '__main__':

    train_batch_size = 32
    eval_batch_size = 64
    sequence_length = 64
    learning_rate = 2e-5
    train_data_size = 368624  # 368624
    eval_data_size = 52661 # 52661
    steps_per_epoch = train_data_size // train_batch_size
    epochs = 1
    warmup_steps = int(epochs * train_data_size * 0.1 / train_batch_size)
    eval_steps = int(math.ceil(eval_data_size / eval_batch_size))
    num_classes = 2
    log_steps = 1
    model_dir = "results/classifier/1/"
    bert_config_file = "/home/geb/PycharmProjects/bert/vocab_file/bert_config.json"
    checkpoint_file = "/home/geb/PycharmProjects/bert/vocab_file/bert_model.ckpt"
    # bert_config_file = "/home/geb/PycharmProjects/bert_ngc/vocab_file/albert_zh/bert_config.json"
    # checkpoint_file = "/home/geb/PycharmProjects/bert_ngc/vocab_file/albert_zh/bert_model.ckpt"

    checkpoint_path = "results/classifier/checkpoint-{:02d}".format(epochs)
    saved_model_path = "saved_models/{}".format(int(time.time()))
    train = True
    predict = False
    export = False

    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy='one_device',
        num_gpus=1,
        tpu_address=False)

    with strategy.scope():
        if train:
            # load data
            train_data = create_classifier_dataset("tf_records/sentence_classifier/train.record0", sequence_length,
                                                   train_batch_size)

            dev_data = create_classifier_dataset("tf_records/sentence_classifier/dev.record0", sequence_length,
                                                 eval_batch_size, False)

            bert_config = BertConfig.from_json_file(bert_config_file)

            cls = BertClassifier(bert_config, sequence_length, num_classes)

            cls.init_pre_training_weights(checkpoint_file)

            optimizer = get_optimizer(learning_rate, steps_per_epoch, epochs, warmup_steps)
            loss_fn = get_loss_fn(num_classes)
            callbacks = get_callbacks(train_batch_size, log_steps, model_dir)

            cls.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn()])

            print(cls._encoder_layer.get_layer("transformer/layer_0").get_weights()[0])

            cls.fit(
                train_data,
                validation_data=dev_data,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_steps=eval_steps,
                callbacks=callbacks)


            tf.keras.models.save_model(cls, saved_model_path, save_format='tf')

            print(cls._encoder_layer.get_layer("transformer/layer_0").get_weights()[0])
        elif export:
            bert_config = AlbertConfig.from_json_file(bert_config_file)

            cls = BertClassifier(bert_config, sequence_length, num_classes)
            cls.load_weights(checkpoint_path)
            cls.predict({"input_word_ids": tf.ones([1, 64]), "input_mask": tf.ones([1, 64]), "input_type_ids": tf.zeros([1, 64])})
            tf.keras.models.save_model(cls, saved_model_path, save_format='tf')

        elif predict:
            bert_config = AlbertConfig.from_json_file(bert_config_file)
            cls = BertClassifier(bert_config, sequence_length, num_classes)
            cls.load_weights(checkpoint_path)
            # TODO ...