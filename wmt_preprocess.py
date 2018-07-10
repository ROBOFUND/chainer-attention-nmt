import argparse
import collections
import io
import progressbar
import re

from utils import count_lines
import s3utils

split_pattern = re.compile(r'([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')


def split_sentence(s, use_lower):
    if use_lower:
        s = s.lower()
    s = s.replace('\u2019', "'")
    s = digit_pattern.sub('0', s)
    words = []
    for word in s.strip().split():
        words.extend(split_pattern.split(word))
    words = [w for w in words if w]
    return words


def read_file(path, use_lower):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    with s3utils.get_s3file(path) as f:
        for line in bar(f, max_value=n_lines):
            words = split_sentence(line, use_lower)
            yield words


def preprocess_dataset(path, outpath, vocab_path=None, vocab_size=None,
                       use_lower=False):
    token_count = 0
    counts = collections.Counter()
    with s3utils.open_buf() as f:
        for words in read_file(path, use_lower):
            line = ' '.join(words)
            f.write(line)
            f.write('\n')
            if vocab_path is not None:
                for word in words:
                    counts[word] += 1
            token_count += len(words)
        s3utils.write_obj(outpath, f)
    print('number of tokens: %d' % token_count)

    if vocab_path and vocab_size:
        vocab = [word for (word, _) in counts.most_common(vocab_size)]
        with s3utils.open_buf() as f:
            for word in vocab:
                f.write(word)
                f.write('\n')
            s3utils.write_obj(vocab_path, f)


def main(args):
    preprocess_dataset(
        args.INPUT,
        args.OUTPUT,
        vocab_path=args.vocab_file,
        vocab_size=args.vocab_size,
        use_lower=args.lower
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', help='path to input')
    parser.add_argument('OUTPUT', help='path to input')
    parser.add_argument('--vocab-file', help='vocabulary file to save')
    parser.add_argument('--vocab-size', type=int, default=30000,
                        help='vocabulary file to save')
    parser.add_argument('--lower', action='store_true',
                        help='use lower case')
    args = parser.parse_args()

    main(args)
