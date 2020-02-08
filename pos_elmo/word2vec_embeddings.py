"""
    Word2Vec

    Functionality:
        - read pre-trained vectors from file
        - generate vectors from text
        - provide functionality to convert words to vectors and vice versa

    Ideas for improvements:
        - use specific text source for training
        - use different pre-trained source

    Credits:
        - https://code.google.com/archive/p/word2vec/
        - https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
        - https://rare-technologies.com/word2vec-tutorial/#app

    Q&A:

"""

from gensim.models.keyedvectors import KeyedVectors
import gensim.models
import gensim.downloader as api


class Word2VecPreprocessor:

    def __init__(self):
        self.model = None

    def prepare_pre_trained_data(self, datasource):
        self.model = api.load(datasource)

    def word2vec(self, word):
        return self.model[word]

    def vec2word(self, vec):
        return self.model.most_similar(positive=[vec], topn=1)

    def training_file_iter(self, training_file):
        with open(training_file, "r") as file:
            for line in file:
                yield line.split()

    def train_model(self, training_file):
        self.model = gensim.models.Word2Vec(sentences=self.training_file_iter(training_file),
                                            iter=10,  # number of training iterations, default: 5
                                            workers=4,  # number of parallel workers, default: 1
                                            size=200,  # NN size, default: 100
                                            min_count=5)  # ignore words with < min_count, default: 1

    def store_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = gensim.models.Word2Vec.load(path)


if __name__ == "__main__":
    _DATA_SOURCE = "word2vec-google-news-300"
    # instantiate preprocessor
    preprocessor = Word2VecPreprocessor()
    # download pre-trained data
    preprocessor.prepare_pre_trained_data(_DATA_SOURCE)
