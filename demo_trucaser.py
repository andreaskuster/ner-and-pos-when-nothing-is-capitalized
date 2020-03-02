from datasets.brown.brown import Brown
from truecase.external_utils import predict_truecasing

dataset = Brown()
dataset = dataset.load_data_lowercase()
dataset_sentences = dataset[0]
truecased_senteces = predict_truecasing(dataset_sentences)
