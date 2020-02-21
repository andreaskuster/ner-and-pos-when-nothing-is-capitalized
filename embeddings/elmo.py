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


class ELMo:

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
    # get embedding for "Hello World"
    print("embedding (v1) for \"Hello World\" is {}".format(ELMo.embedding([["Hello", "World"]], [2])))
