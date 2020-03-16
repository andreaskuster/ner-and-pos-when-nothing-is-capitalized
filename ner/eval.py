# coding=utf-8
from __future__ import print_function
import optparse
import torch
import time
import _pickle as cPickle
from torch.autograd import Variable

from loader import *
from utils import *

t = time.time()

# python -m visdom.server

optparser = optparse.OptionParser()
optparser.add_option(
    "-t", "--test", default="dataset/eng.testb",
    # "-t", "--test", default="dataset/twitter.json",
    help="Test set location"
)
optparser.add_option(
    "-t", "--case", default="0",
    # "-t", "--test", default="dataset/twitter.json",
    type='int', help="0 for cased, 1 for lower cased and 2 for truecased"
)

optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    '--model_path', default='models/model',
    help='model path'
)
optparser.add_option(
    '--map_path', default='models/mapping.pkl',
    help='model path'
)


opts = optparser.parse_args()[0]

mapping_file = opts.map_path

with open(mapping_file, 'rb') as f:
    mappings = cPickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

use_gpu =  opts.use_gpu == 1 and torch.cuda.is_available()


assert os.path.isfile(opts.test)
assert parameters['tag_scheme'] in ['iob', 'iobes']

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)

case = parameters['case']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# test_sentences = load_twitter_sentences(opts.test)
# if case==0 or case==1:
test_sentences = load_sentences(opts.test, case, zeros)
# else:
    # test_sentences = load_sentences(opts.test, 1, zeros)

# print(test_sentences[:10])
update_tag_scheme(test_sentences, tag_scheme)
if case==0 or case==1:
    test_data = prepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id, case
    )
else:
    test_data = prepare_dataset_truecase(
        test_sentences, word_to_id, char_to_id, tag_to_id, case
    )    

model = torch.load(opts.model_path)
model_name = opts.model_path.split('/')[-1].split('.')[0]

if use_gpu:
    model.cuda()
print("evaluating")
model.eval()


def eval(model, datas, maxl=1):
    prediction = []
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        # l = getmaxlen(ground_truth_id)
        # if not l == maxl:
        #     continue
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']


        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(),chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    predf = eval_temp + '/pred.' + model_name
    scoref = eval_temp + '/score.' + model_name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    with open(scoref, 'rb') as f:
        for l in f.readlines():
            print(l.strip())

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))


eval(model, test_data, 1)

print(time.time() - t)
