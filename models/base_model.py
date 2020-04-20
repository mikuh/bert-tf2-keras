import tensorflow as tf
from utils import load_from_google_albert, load_from_google_bert, get_transformer_encoder
from configs import AlbertConfig, BertConfig


class BaseModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_transformer_encoder(self, bert_config, sequence_length):
        """get transformer encoder model
        """
        return get_transformer_encoder(bert_config, sequence_length)

    def init_pre_training_weights(self, checkpoint_file):
        """init bert weights from pre training checkpoint
        """
        if isinstance(self.bert_config, AlbertConfig):
            load_from_google_albert(self, checkpoint_file)
        else:
            load_from_google_bert(self, checkpoint_file)
