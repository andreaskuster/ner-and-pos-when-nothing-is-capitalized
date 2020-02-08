"""
    Credits:
        - https://nlp.stanford.edu/projects/glove/
        - https://github.com/stanfordnlp/GloVe
"""


import glove


model = glove.Glove(cooccur, d=50, alpha=0.75, x_max=100.0)

for epoch in range(25):
    err = model.train(batch_size=200, workers=4, d=50)
    print("epoch %d, error %.3f" % (epoch, err), flush=True)




if __name__ == "__main__":
    pass



