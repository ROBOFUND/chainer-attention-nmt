"""Train a seq2seq model."""
import argparse

import numpy
import six

import chainer
from chainer import training
from chainer.training import extensions

from net import Seq2seq
from metrics import CalculateBleu
from utils import load_vocabulary
from utils import load_data
from utils import calculate_unknown_ratio
from utils import seq2seq_pad_concat_convert

from natto import MeCab
np = numpy
UNK = 0
EOS = 1

def main():
    parser = argparse.ArgumentParser(description='Attention-based NMT')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('model_npz', help='model file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--encoder-unit', type=int, default=128,
                        help='number of units')
    parser.add_argument('--encoder-layer', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--encoder-dropout', type=int, default=0.1,
                        help='number of layers')
    parser.add_argument('--decoder-unit', type=int, default=128,
                        help='number of units')
    parser.add_argument('--attention-unit', type=int, default=128,
                        help='number of units')
    parser.add_argument('--maxout-unit', type=int, default=128,
                        help='number of units')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--debug', action='store_true',
                        help='use a small part of training data')
    args = parser.parse_args()

    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    model = Seq2seq(len(source_ids), len(target_ids), args.encoder_layer,
                    args.encoder_unit, args.encoder_dropout,
                    args.decoder_unit, args.attention_unit, args.maxout_unit)
    chainer.serializers.load_npz(args.model_npz, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    m = MeCab('-Owakati')
    while True:
        line = input('> ')
        words = m.parse(line).split()
        words.append('<EOS>')
        x = np.zeros((1, len(words)), dtype=np.int32)
        for i in range(len(words)):
            x[0, i] = source_ids.get(words[i], UNK)
        result = model.translate(x)
        o_words = []
        for i in range(len(result[0])):
            o_words.append(target_words.get(result[0][i], '<unk>'))
            if o_words[-1] == '<EOS>':
                o_words.pop()
                break
        print(" ".join(o_words))
        #import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
