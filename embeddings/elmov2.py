from allennlp.modules.elmo import Elmo, batch_to_ids


class ELMo:

    def __init__(self):
        self.options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        self.weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.elmo = Elmo(self.options_file,
                         self.weight_file, 2,
                         dropout=0)
        self.dim = 1024

    def word2vec(self, word):
        return self.elmo(batch_to_ids(word))["elmo_representations"]

    def embedding(self, sentences):
        # use batch_to_ids to convert sentences to character ids
        # sentences = [['First', 'sentence', '.'], ['Another', '.']]
        character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)["elmo_representations"][0]

        return embeddings.detach().numpy()


if __name__ == "__main__":
    # instantiate class
    elmov2 = ELMo()
    # get embedding for "Hello World"
    print("embedding (v2) for \"Hello World\" is {}".format([elmov2.word2vec(word) for word in ["Hello", "World"]]))
