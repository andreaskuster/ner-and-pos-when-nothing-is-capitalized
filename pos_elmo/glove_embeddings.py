"""
    GloVe: Global Vectors for Word Representation

    Functionality:
        - read pre-trained vectors from file
        - generate vectors from text
        - provide functionality to convert words to vectors and vice versa

    Ideas for improvements:
        - use specific text source for training
        - use different pre-trained source

    Credits:
        - https://nlp.stanford.edu/projects/glove/
        - https://github.com/stanfordnlp/GloVe
        - https://github.com/JonathanRaiman/glove
        - https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
        - https://medium.com/ai-society/jkljlj-7d6e699895c4
        - https://linguistics.stackexchange.com/questions/3641/how-to-calculate-the-co-occurrence-between-two-words-in-a-window-of-text

    Q&A:
        How to install glove: pip3 install https://github.com/JonathanRaiman/glove/archive/master.zip
        Co-occurrence matrix format:

            cooccurr = {
                0: {
                    0: 1.0,
                    2: 3.5
                },
                1: {
                    2: 0.5
                },
                2: {
                    0: 3.5,
                    1: 0.5,
                    2: 1.2
                }
            }
"""

import glove
import zipfile
import urllib.request
import os
import numpy as np


class GlovePreprocessor:

    def __init__(self):
        self.data_sources = {
            "wikipedia2014_gigaword5_6b_uncased": {
                "url": "http://nlp.stanford.edu/data/glove.6B.zip",
                "files": [{
                        "name": "glove.6B.50d.txt",
                        "dim": 50
                    },
                    {
                        "name": "glove.6B.100d.txt",
                        "dim": 100
                    },
                    {
                        "name": "glove.6B.200d.txt",
                        "dim": 200
                    },
                    {
                        "name": "glove.6B.300d.txt",
                        "dim": 300
                    }
                ]
            },
            "common_crawl_42b_uncased": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
            "common_crawl_840b_cased": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
            "twitter_2b_uncased": "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
        }
        self.base_path = "data"
        self.embeddings = dict()
        self.co_occur = dict()
        self.word2no = dict()
        self.no2word = list()
        self.counter = 0
        self.model = None
        self.pre_trained = None

    def prepare_pre_trained_data(self, datasource, dataset):
        # check if file already exists
        if not os.path.isfile(os.path.join(self.base_path, datasource, dataset)):
            # check if zip file exits
            if not os.path.isfile(os.path.join(self.base_path, "{}.zip".format(datasource))):
                self.download_data(datasource)
            # unzip file
            self.unzip_data(datasource, dataset)

    def download_data(self, datasource):
        # get data from data source
        urllib.request.urlretrieve(self.data_sources[datasource]["url"], os.path.join(self.base_path, "{}.zip".format(datasource)))

    def unzip_data(self, datasource, dataset):
        zip_ref = zipfile.ZipFile(os.path.join(self.base_path, "{}.zip".format(datasource)), "r")
        # print("files in zip: {}".format(zip_ref.namelist()))
        os.mkdir(os.path.join(self.base_path, datasource))
        zip_ref.extractall(os.path.join(self.base_path, datasource))
        zip_ref.close()

    def import_pre_trained(self, datasource, dataset):
        self.embeddings = dict()
        with open(os.path.join(self.base_path, datasource, dataset), "r") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings[word] = vector
        self.pre_trained = True

    def token_iter(self, path):
        with open(path, "r") as file:
            for line in file:
                yield line.split()

    def word_to_no(self, word):
        if word in self.word2no:
            return self.word2no[word]
        else:
            self.word2no[word] = self.counter
            self.no2word.append(word)
            self.counter += 1
            return self.word2no[word]

    def create_co_occurrence_matrix(self, path, sliding_window_size=3):
        window = list()
        iter = self.token_iter(path)
        self.co_occur = dict()
        # fill initial window
        for i in range(sliding_window_size):
            window.append(iter.__next__())
        # process all windows
        for item in iter.__next__():
            # add entries to matrix
            center_pos = np.ceil(sliding_window_size/2) + 1
            center = window[center_pos]
            for word in window[:center_pos-1] + window[center_pos+1:]:
                if center not in self.co_occur:
                    self.co_occur[self.word_to_no(center)] = dict()
                self.co_occur[self.word2no(center)][self.word2no(word)] += 1
            # new windows
            window = window[1:].append(item)

    def train_model(self):
        self.model = glove.Glove(self.co_occur, d=50, alpha=0.75, x_max=100.0)
        for epoch in range(100):
            err = self.model.train(batch_size=200, workers=8)
            print("epoch %d, error %.3f" % (epoch, err), flush=True)
        self.pre_trained = False

    def word2vec(self, word):
        if self.pre_trained:
            return self.embeddings[word]
        elif self.pre_trained == False:
            return self.model.W[self.word2no[word]]
        else:
            raise RuntimeError("No vectors trained.")

    def vec2word(self, vec):
        raise NotImplementedError()


if __name__ == "__main__":
    _DATA_SOURCE = "wikipedia2014_gigaword5_6b_uncased"
    _DATA_SET = "glove.6B.100d.txt"
    # instantiate preprocessor
    preprocessor = GlovePreprocessor()
    # prepare pre-trained data
    preprocessor.prepare_pre_trained_data(_DATA_SOURCE, _DATA_SET)
    # import data
    preprocessor.import_pre_trained(_DATA_SOURCE, _DATA_SET)

