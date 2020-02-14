"""

    Penn TreeBank

    File placement:
        - add penn treebank wsj files into ~/nltk_data/corpora/treebank/combined (extracted from 00-24 structure)
        - do once: execute generate_file_structure() over the initial ptb file setup (in folder 00,.. 24)

    Credits:
    - http://www.nltk.org/howto/corpus.html#parsed-corpora
        - https://www.sketchengine.eu/penn-treebank-tagset/

"""

import os
import json
import nltk


class PTB:

    def __init__(self):
        # load config from file
        with open(os.path.join(os.path.dirname(__file__), "ptb_conf.json")) as config:
            self.config = json.load(config)

    @staticmethod
    def generate_file_structure(base_path):
        # print file structure of ptb dataset, for the config
        for i in range(25):
            train_files = [item for item in os.listdir(os.path.join("{}/wsj/{:02}".format(base_path, i)))]
            print("\"{}\":{},".format(i, train_files))

    def load_data(self, sections, text_map_func=lambda x: x, tag_map_func=lambda x: x):
        # get all files
        files = list()
        for i in sections:
            files += self.config["files"][str(i)]
        # split text from pos tag
        text, tag = [], []
        for tagged_sentence in nltk.corpus.treebank.tagged_sents(files):
            sentence, tags = zip(*tagged_sentence)
            text.append(list(map(text_map_func, sentence)))
            tag.append(list(map(tag_map_func, tags)))
        # return as tuple
        return text, tag

    def load_data_lowercase(self, sections):
        return self.load_data(sections=sections,
                              text_map_func=str.lower)


if __name__ == "__main__":
    # instantiate class
    ptb = PTB()
    # get ptb section 0 and 1
    ptb01 = ptb.load_data([0, 1])
    # get ptb section 7 and 10, lowercase
    ptb710_lower = ptb.load_data_lowercase([7, 10])
