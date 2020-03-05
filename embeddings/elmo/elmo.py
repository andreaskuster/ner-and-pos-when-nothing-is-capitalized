#!/usr/bin/env python3
# encoding: utf-8

"""
    Copyright (C) 2020  Andreas Kuster

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Andreas Kuster"
__copyright__ = "Copyright 2020"
__license__ = "GPL"

import tensorflow_hub as hub

"""
    ELMo: Deep contextualized word representations

    Credits:
        - https://jalammar.github.io/illustrated-bert/
        - https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440
        - https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/
        - https://allennlp.org/elmo
        - https://tfhub.dev/google/elmo/3

"""


class ELMo:

    @staticmethod
    def embedding(tokens_input, tokens_length):

        # load model from tensorflow hub
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

        # get embeddings
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
    sentences = [["Hello", "World"]]
    token_lengths = list(map(len, sentences))
    print("embedding (v1) for \"Hello World\" is {}".format(ELMo.embedding(sentences, token_lengths)))
