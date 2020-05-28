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

import gensim.models
import gensim.downloader as api

import numpy as np

from typing import List

"""
    Word2Vec

    Credits:
        - https://code.google.com/archive/p/word2vec/
        - https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
        - https://rare-technologies.com/word2vec-tutorial/#app
"""


class Word2Vec:

    def __init__(self):
        self.model = None  # word2vec model instance
        self.dim = 300  # embedding vector dimension

    def import_pre_trained_data(self, datasource: str = "word2vec-google-news-300"):
        print("Downloading and importing {}, this might take a while...".format(datasource))
        # load pre-trained model from file
        self.model = api.load(datasource)
        print("Embedding import finished")

    def word2vec(self, word: str) -> List[float]:
        # return word vector if the word exists, return the zero vector otherwise
        if word in self.model:
            return self.model[word]
        else:
            return np.zeros(self.dim)

    def vec2word(self, vec: List[float]) -> str:
        # return closest word (by cosine similarity)
        return self.model.most_similar(positive=[vec], topn=1)

    def training_file_iter(self, training_file: str):
        # iterate over the tokens, without reading the whole file at once (for memory efficiency)
        with open(training_file, "r") as file:
            for line in file:
                yield line.split()

    def train_model(self, training_file: str):
        # train the model
        self.model = gensim.models.Word2Vec(sentences=self.training_file_iter(training_file),
                                            iter=10,  # number of training iterations, default: 5
                                            workers=4,  # number of parallel workers, default: 1
                                            size=200,  # NN size, default: 100
                                            min_count=5)  # ignore words with < min_count, default: 1

    def store_model(self, path: str):
        # save trained model
        self.model.save(path)

    def load_model(self, path: str):
        # load trained model
        self.model = gensim.models.Word2Vec.load(path)


if __name__ == "__main__":
    _DATA_SOURCE = "word2vec-google-news-300"
    # instantiate preprocessor
    preprocessor = Word2Vec()
    # download pre-trained data
    preprocessor.import_pre_trained_data(_DATA_SOURCE)
    # get embedding for "Hello World"
    print("embedding for \"Hello World\" is {}".format([preprocessor.word2vec(word) for word in ["Hello", "World"]]))
