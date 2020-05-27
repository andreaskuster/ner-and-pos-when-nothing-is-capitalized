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

import os
import json
import nltk
import random

from typing import Tuple, List

from truecase.external_utils import predict_truecasing
from datasets import AbstractLoader

"""

    Penn TreeBank

    File placement:
        - add all penn treebank .wsj files into ~/nltk_data/corpora/treebank/combined (extracted from 00-24 structure)
        - do once: execute generate_file_structure() over the initial ptb file setup (in folder 00,.. 24) or
          use ptb_conf.json

    Credits:
        - http://www.nltk.org/howto/corpus.html#parsed-corpora
        - https://www.sketchengine.eu/penn-treebank-tagset/

"""


class PTB(AbstractLoader):

    def __init__(self):
        # initialize super class
        super().__init__()
        # load config from file
        with open(os.path.join(os.path.dirname(__file__), "ptb_conf.json")) as config:
            self.config = json.load(config)
        # download public available subset of penn treebank data
        if not os.path.isfile("~/nltk_data/corpora/treebank.zip"):
            nltk.download("treebank")

    @staticmethod
    def generate_file_structure(base_path):
        # print file structure of ptb dataset, for the config
        for i in range(25):
            train_files = [item for item in os.listdir(os.path.join("{}/wsj/{:02}".format(base_path, i)))]
            print("\"{}\":{},".format(i, train_files))

    def load_data(self, sections=list(range(25)), text_map_func=lambda x: x, tag_map_func=lambda x: x) -> Tuple[List, List]:
        # get all files
        files = list()
        for i in sections:
            files += self.config["files"][str(i)]
        # split text from pos tag
        text, tag = list(), list()
        for tagged_sentence in nltk.corpus.treebank.tagged_sents(files):
            sentence, tags = zip(*tagged_sentence)
            text.append(list(map(text_map_func, sentence)))
            tag.append(list(map(tag_map_func, tags)))
        # return as tuple
        return text, tag

    def load_data_lowercase(self, sections) -> Tuple[List, List]:
        # apply lowercase function to the dataset
        return self.load_data(sections=sections,
                              text_map_func=str.lower)

    def load_data_truecase(self, sections) -> Tuple[List, List]:
        # fetch lowercase dataset and truecase it
        lower_sentence, tag = self.load_data_lowercase(sections)
        return predict_truecasing(lower_sentence), tag

    def load_data_cased_and_uncased(self, sections) -> Tuple[List, List]:
        # fetch cased and uncased dataset and concatenate them
        sentence_c, tag_c = self.load_data(sections=sections)
        sentence_u, tag_u = self.load_data_lowercase(sections=sections)
        return sentence_c + sentence_u, tag_c + tag_u

    def load_data_half_mixed(self, sections) -> Tuple[List, List]:
        # fetch cased dataset
        sentence, tag = self.load_data(sections=sections)
        # generate 50% random indices from 0..len(sentence)-1
        rand_samples = random.sample(range(0, len(sentence)), int(0.5 * len(sentence)))
        # lowercase the elements at the address of the indices
        for index in rand_samples:
            sentence[index] = list(map(str.lower, sentence[index]))
        return sentence, tag


if __name__ == "__main__":
    # instantiate class
    ptb = PTB()
    # get ptb section 0 and 1
    ptb01 = ptb.load_data([0, 1])
    # get ptb section 7 and 10, lowercase
    ptb710_lower = ptb.load_data_lowercase([7, 10])
    # get full ptb dataset, truecase
    data_truecase = ptb.load_data_truecase([2])
    # get c+u ptb data (full dataset cased and uncased)
    data_cu = ptb.load_data_cased_and_uncased([3, 4])
    # get hm ptb data (50% randomly lowercase)
    data_hm = ptb.load_data_half_mixed([5, 6, 7, 8])
