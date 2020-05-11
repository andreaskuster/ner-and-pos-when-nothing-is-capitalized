from truecase.external_utils import predict_truecasing

import os


class CoNLL2003:

    def __init__(self, train_path=None, test_path=None):
        absFilePath = os.path.abspath(__file__)
        folder_path, _ = os.path.split(absFilePath)
        if train_path is None:
            self.train_path = os.path.join(folder_path, 'eng.train')
        else:
            self.train_path = train_path
        if test_path is None:
            self.test_path = os.path.join(folder_path, 'eng.testb')
        else:
            self.test_path = test_path

        self.train_words = []
        self.train_tags = []
        self.test_words = []
        self.test_tags = []

    @staticmethod
    def __load_file(file, text_map_func=lambda x: x, tag_map_func=lambda x: x):
        result_words = []
        result_tags = []

        curr_sentence_words = []
        curr_sentence_tags = []

        for line in file.readlines()[2:]:
            line = line.split()
            if len(line) == 0:
                result_words.append(curr_sentence_words)
                result_tags.append(curr_sentence_tags)

                curr_sentence_words = []
                curr_sentence_tags = []
            else:
                curr_sentence_words.append(text_map_func(line[0]))
                curr_sentence_tags.append(tag_map_func(line[1:]))
        return result_words, result_tags

    def load_data(self, text_map_func=lambda x: x, tag_map_func=lambda x: x):
        if len(self.train_words) == 0:
            with open(self.train_path) as f:
                self.train_words, self.train_tags = CoNLL2003.__load_file(f, text_map_func, tag_map_func)
            with open(self.test_path) as f:
                self.test_words, self.test_tags = CoNLL2003.__load_file(f, text_map_func, tag_map_func)

        return (self.train_words, self.train_tags), (self.test_words, self.test_tags)

    def load_data_lowercase(self):
        return self.load_data(text_map_func=str.lower)

    def load_data_truecase(self):
        (train_lower, train_tag), (test_lower, test_tag) = self.load_data_lowercase()
        return (predict_truecasing(train_lower), train_tag), (predict_truecasing(test_lower), test_tag)


if __name__ == "__main__":
    # instantiate class
    conll2003 = CoNLL2003()
    # get conll2003 data
    data = conll2003.load_data()
    # get conll2003 data, lowercase
    data_lower = conll2003.load_data_lowercase()
