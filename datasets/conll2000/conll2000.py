import nltk
from truecase.external_utils import predict_truecasing


class CoNLL2000:

    def __init__(self):
        # download dataset
        nltk.download("conll2000")

    def load_data(self, text_map_func=lambda x: x, tag_map_func=lambda x: x):
        # split text from pos tag
        text, tag = [], []
        for tagged_sentence in nltk.corpus.conll2000.tagged_sents():
            sentence, tags = zip(*tagged_sentence)
            text.append(list(map(text_map_func, sentence)))
            tag.append(list(map(tag_map_func, tags)))
        # return as tuple
        return text, tag

    def load_data_lowercase(self):
        return self.load_data(text_map_func=str.lower)

    def load_data_truecase(self):
        lower_sentence, tag = self.load_data_lowercase()
        return predict_truecasing(lower_sentence), tag

    
if __name__ == "__main__":
    # instantiate class
    conll2000 = CoNLL2000()
    # get conll2000 data
    data = conll2000.load_data()
    # get conll2000 data, lowercase
    data_lower = conll2000.load_data_lowercase()
