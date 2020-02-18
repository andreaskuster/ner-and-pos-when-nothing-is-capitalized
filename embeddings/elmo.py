"""
    ELMo: Deep contextualized word representations

    Functionality:
        - read pre-trained vectors from file
        - generate vectors from text
        - provide functionality to convert words to vectors and vice versa

    Ideas for improvements:
        - use specific text source for training
        - use different pre-trained source

    Author: Andreas

    Credits:
        - https://jalammar.github.io/illustrated-bert/
        - https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440
        - https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/
        - https://allennlp.org/elmo
        - https://tfhub.dev/google/elmo/3

    Q&A:

"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids


class ELMoV2:

    def __init__(self):
        self.options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        self.weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.elmo = Elmo(self.options_file,
                         self.weight_file, 2,
                         dropout=0)
        self.dim = 1024

    def word2vec(self, word):
        return self.elmo(batch_to_ids(word))["elmo_representations"]

    def embedding(self, sentences):

        # use batch_to_ids to convert sentences to character ids
        # sentences = [['First', 'sentence', '.'], ['Another', '.']]
        character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)["elmo_representations"][0]

        return embeddings.detach().numpy()


class ELMoV1:

    @staticmethod
    def embedding(tokens_input, tokens_length):

        sess = tf.compat.v1.Session()

        from tensorflow.python.keras.backend import set_session
        set_session(sess)

        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        embeddings = elmo(
            inputs={
                "tokens": tokens_input,
                "sequence_len": tokens_length
            },
            signature="tokens",
            as_dict=True)["elmo"]
        return embeddings


if __name__ == "__main__":
    # instantiate class
    elmov2 = ELMoV2()
    # get embedding for "Hello World"
    print("embedding (v1) for \"Hello World\" is {}".format(ELMoV1.embedding([["Hello", "World"]], [2])))
    print("embedding (v2) for \"Hello World\" is {}".format([elmov2.word2vec(word) for word in ["Hello", "World"]]))