# Truecasing

## Introduction
This part is related to the truecasing part of the paper. Each file has it's specific purpose, but for outside use (what ner and pos are using) `external_utils.py` and `__init__.py` are crucial, since they allow user to just give a corpus, and have it truecased.

An important note is that all files in this directory should be called from this directory (i.e. after `cd truecase`).

## Result
Due to mistake in way OOV was done initially (look at `TrueCaseDataset` vs. `TrueCaseDatasetImproved` in `singlechar_dataset.py`) there are 2 versions of results. A standard one, and `_old_oov`.

For any use purpose a one without suffix should be used, though there haven't been any performance differences observed between the two.

`token_to_idx.pkl` is just a mapping from a character to an idx corresponding to a unit vector.

`model_our.pkl` is a BiLSTM model, as described in the Susanto Paper (referenced in the root repo README).

## Files and their purposes

* `external_utils` - File providing one, simple to use function, which can be used by any external agents (such as ner and pos) to use the truecaser.
* `singlechar_dataset` - File consisting of 2 PyTorch Dataset Classes both implementing 2 character level dataset, with labels being 1 (is uppercase) or 0 (is lowercase/uppercase undefined). Also contains a function to load a dataset given token to index matchings (used for validation/test).
* `train_truecaser` - Main file, where truecaser is trained, and a few results after are plotted. Should not be imported since it calculates a few values modifying data on the fly (moving to GPU etc.). To reproduce results, should be as easy as calling `python train_truecaser.py` in this directory (up to commenting/uncommenting lines 180-182).
* `truecase_model` - PyTorch Module (aka Model) representing one described in Susanto's paper. Also contains a easy-to-use function that loads model with pre-trained weights.
* `truecase_table_generate` - File for final paper specifically. Runs Truecaser on CoNLL2003, PTB datasets, and returns results.

## External Links
[Github Link to Susanto Paper (the OG implementation)](https://github.com/raymondhs/char-rnn-truecase)