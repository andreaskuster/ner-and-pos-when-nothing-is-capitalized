"""
    GloVe: Global Vectors for Word Representation

    Functionality:
        - read pre-trained vectors from file
        - generate vectors from text
        - provide functionality to convert words to vectors and vice versa

    Ideas for improvements:
        - use specific text source for training
        - use different pre-trained source

    Author: Andreas

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


class GloVe:

    def __init__(self):
        self.data_sources = {
            "wikipedia2014_gigaword5_6b_uncased": {
                "url": "http://nlp.stanford.edu/data/glove.6B.zip",
                "glove.6B.50d.txt": {
                    "dim": 50
                },
                "glove.6B.100d.txt": {
                    "dim": 100
                },
                "glove.6B.200d.txt": {
                    "dim": 200
                },
                "glove.6B.300d.txt": {
                    "dim": 300
                }
            },
            "common_crawl_42b_uncased": {
                "url": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
                "glove.42B.300d": {
                    "dim": 300
                }
            },
            "common_crawl_840b_cased": {
                "url": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
                "glove.840B.300d.txt": {
                    "dim": 300
                }
            },
            "twitter_2b_uncased": {
                "url": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
                "glove.twitter.27B.25d.txt": {
                    "dim": 25
                },
                "glove.twitter.27B.50d.txt": {
                    "dim": 50
                },
                "glove.twitter.27B.100d.txt": {
                    "dim": 100
                },
                "glove.twitter.27B.200d.txt": {
                    "dim": 200
                }
            }
        }
        self.base_path = "../embeddings/data"
        self.embeddings = dict()
        self.co_occur = dict()
        self.word2no = dict()
        self.no2word = list()
        self.counter = 0
        self.model = None
        self.pre_trained = None
        self.dim = 0

    def import_pre_trained_data(self, datasource, dataset):
        print("Importing pre-trained data, this might take a while...")
        # check if file already exists
        if not os.path.isfile(os.path.join(self.base_path, datasource, dataset)):
            # check if zip file exits
            if not os.path.isfile(os.path.join(self.base_path, "{}.zip".format(datasource))):
                self.download_data(datasource)
            # unzip file
            self.unzip_data(datasource, dataset)
        # import data
        self.import_pre_trained(datasource, dataset)
        print("Embedding import finished")

    def download_data(self, datasource):
        # check if folder exists
        if not os.path.isdir(self.base_path):
            # create folder
            os.mkdir(self.base_path)
        # get data from data source
        print("Downloading {}, this might take a while...".format(datasource))
        urllib.request.urlretrieve(self.data_sources[datasource]["url"],
                                   os.path.join(self.base_path, "{}.zip".format(datasource)))

    def unzip_data(self, datasource, dataset):
        zip_ref = zipfile.ZipFile(os.path.join(self.base_path, "{}.zip".format(datasource)), "r")
        os.mkdir(os.path.join(self.base_path, datasource))
        print("Unzip {}, this might take a while...".format(datasource))
        zip_ref.extractall(os.path.join(self.base_path, datasource))
        zip_ref.close()

    def import_pre_trained(self, datasource, dataset):
        # set embedding vector dimensionality
        self.dim = self.data_sources[datasource][dataset]["dim"]
        print("Importing embedding {}, this might take a while...".format(dataset))
        self.embeddings = dict()
        with open(os.path.join(self.base_path, datasource, dataset), "r") as f:
            for line in f:
                values = line.split(" ")
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
            center_pos = np.ceil(sliding_window_size / 2) + 1
            center = window[center_pos]
            for word in window[:center_pos - 1] + window[center_pos + 1:]:
                if center not in self.co_occur:
                    self.co_occur[self.word_to_no(center)] = dict()
                self.co_occur[self.word2no(center)][self.word2no(word)] += 1
            # new windows
            window = window[1:].append(item)

    def train_model(self):
        self.dim = 50
        self.model = glove.Glove(self.co_occur, d=self.dim, alpha=0.75, x_max=100.0)
        for epoch in range(100):
            err = self.model.train(batch_size=200, workers=8)
            print("epoch %d, error %.3f" % (epoch, err), flush=True)
        self.pre_trained = False

    def word2vec(self, word):
        if self.pre_trained:
            if word in self.embeddings:
                return self.embeddings[word]
            else:
                return np.zeros(self.dim)
        elif self.pre_trained == False:
            return self.model.W[self.word2no[word]]
        else:
            raise RuntimeError("No vectors trained.")

    def vec2word(self, vec):
        raise NotImplementedError()


if __name__ == "__main__":
    _DATA_SOURCE = "common_crawl_840b_cased"
    _DATA_SET = "glove.840B.300d.txt"
    # instantiate preprocessor
    preprocessor = GloVe()
    # prepare pre-trained data
    preprocessor.import_pre_trained_data(_DATA_SOURCE, _DATA_SET)
    # get embedding for "Hello World"
    print("embedding for \"Hello World\" is {}".format([preprocessor.word2vec(word) for word in ["Hello", "World"]]))
