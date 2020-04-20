import tensorflow as tf
import csv
import numpy as np


def decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def single_file_dataset(input_file, name_to_features):
    """Creates a single-file dataset to be passed for BERT custom training."""
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    d = d.map(lambda record: decode_record(record, name_to_features))

    # When `input_file` is a path to a single file or a list
    # containing a single path, disable auto sharding so that
    # same input file is sent to all workers.
    if isinstance(input_file, str) or len(input_file) == 1:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard = False
        d = d.with_options(options)
    return d


def create_classifier_dataset(file_path,
                              seq_length,
                              batch_size,
                              is_training=True,
                              input_pipeline_context=None):
    """Creates input dataset from (tf)records files for train/eval."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'label_id': tf.io.FixedLenFeature([], tf.int64),
        'is_real_example': tf.io.FixedLenFeature([], tf.int64),
    }
    dataset = single_file_dataset(file_path, name_to_features)

    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)

    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids']
        }
        y = record['label_id']
        return (x, y)

    dataset = dataset.map(_select_data_from_record)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(1024)
    return dataset


def create_squence_label_set(file_path,
                             seq_length,
                             batch_size,
                             is_training=True,
                             input_pipeline_context=None):
    """Creates input dataset from (tf)records files for train/eval."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'is_real_example': tf.io.FixedLenFeature([], tf.int64),
    }
    dataset = single_file_dataset(file_path, name_to_features)
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)

    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids']
        }
        y = record['label_ids']
        return (x, y)

    dataset = dataset.map(_select_data_from_record)
    if is_training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(1024)
    return dataset


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def convert_txt_inputs(sequences: list, max_seq_length: int, tokenizer):
    inputs = {
        "input_word_ids": [],
        "input_mask": [],
        "input_type_ids": []
    }
    for sequence in sequences:
        textlist = list(sequence)
        tokens = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            ntokens.append("**NULL**")
            # label_mask.append(0)
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        inputs["input_word_ids"].append(input_ids)
        inputs["input_mask"].append(input_mask)
        inputs["input_type_ids"].append(segment_ids)
    inputs["input_word_ids"] = np.array(inputs["input_word_ids"])
    inputs["input_mask"] = np.array(inputs["input_mask"])
    inputs["input_type_ids"] = np.array(inputs["input_type_ids"])
    return inputs


if __name__ == '__main__':
    # name_to_features = {
    #     'input_ids': tf.io.FixedLenFeature([36], tf.int64),
    #     'input_mask': tf.io.FixedLenFeature([36], tf.int64),
    #     'segment_ids': tf.io.FixedLenFeature([36], tf.int64),
    #     'label_ids': tf.io.FixedLenFeature([36], tf.int64),
    #     'is_real_example': tf.io.FixedLenFeature([], tf.int64),
    # }
    #
    # d = single_file_dataset(
    #     "/home/geb/PycharmProjects/bert-tf2-keras/tf_records/ner/dev.record0",
    #     name_to_features)
    #
    # for x in d:
    #     print(x)
    #     break

    from utils import tokenization

    tokenizer = tokenization.FullTokenizer("/home/geb/PycharmProjects/bert_ngc/vocab_file/albert_zh/vocab.txt")

    inputs = convert_txt_inputs("我操你妈", 10, tokenizer)
    print(inputs)
