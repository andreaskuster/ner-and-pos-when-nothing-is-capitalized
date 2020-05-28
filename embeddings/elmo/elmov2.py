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

from typing import List

from allennlp.modules.elmo import Elmo, batch_to_ids

from embeddings import AbstractEmbedding


class ELMo(AbstractEmbedding):

    def __init__(self):
        # initialize super class
        super().__init__()
        # weights and definition download urls
        self.options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway" \
                            "/elmo_2x4096_512_2048cnn_2xhighway_options.json "
        self.weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway" \
                           "/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 "
        # instantiate class
        self.elmo = Elmo(options_file=self.options_file,
                         weight_file=self.weight_file,
                         num_output_representations=2,
                         dropout=0)
        # set embedding dimensionality
        self.dim = 1024

    def word2vec(self, word: str) -> List[float]:
        # map the word to its embedding vector
        return self.elmo(batch_to_ids(word))["elmo_representations"]

    def embedding(self, sentences: List[str]) -> List[float]:
        # use batch_to_ids to convert sentences to character ids
        character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)["elmo_representations"][0]
        # return the actual array (convert symbolic stencil to numpy array)
        return embeddings.detach().numpy()


if __name__ == "__main__":
    # instantiate class
    elmov2 = ELMo()
    # get embedding for "Hello World"
    sentence = ["Hello", "World"]
    print("embedding (v2) for \"Hello World\" is {}".format([elmov2.word2vec(word) for word in sentences]))
