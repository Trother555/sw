import re
import unicodedata
import glob
import os
import string
import argparse
import sys

from tqdm import tqdm
import chardet
import nltk


nltk.data.path.append('./nltk_data')


lathin = set(list("<>—-»«“„()@\"\'#№;$%^:*1234567890") +
             list("ABCDEFGHIJKLMNOPQRSTWXZ") +
             list("ABCDEFGHIJKLMNOPQRSTWXZ".lower()))

punkt = string.punctuation + '–'


def count_words(sentence):
    words = nltk.tokenize.word_tokenize(sentence)
    return len([w for w in words if w not in punkt])


def process(corp, res_sents, enc='UTF-8'):
    with open(corp, encoding=enc) as corpus:
        text = corpus.read()

        sents = nltk.sent_tokenize(text)

        # remove all sentences with latin letters
        sents = [s for s in sents if set.isdisjoint(set(s), lathin)]
        # normalize unicode
        # sents = [unicodedata.normalize("NFKD", s) for s in sents]
        # remove \n and [, ]
        sents = [re.sub(r'[\[\]\n]', '', s).strip() for s in sents]
        # remove one-words
        sents = [s for s in sents if count_words(s) > 1]
        # save each sentence on separate line
        with open(res_sents, 'a', encoding='UTF-8') as f:
            for s in sents:
                f.write(f'{s}\n')


def detect_enc(fname):
    with open(fname, 'rb') as f:
        return chardet.detect(f.read(40000))['encoding']


def main(args):
    parser = argparse.ArgumentParser(description='Preprocess text corp.')
    parser.add_argument('--target_dir', help='Target directory containing corpora files in txt', required=True)
    parser.add_argument('--res_dir', help='Directory to save preprocess results', required=True)
    args = parser.parse_args(args)
    target_dir = args.target_dir
    res_dir = args.res_dir
    files = glob.glob(os.path.join(target_dir, '*.txt'))
    print('Processing files')

    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    for file in tqdm(files):
        fname = os.path.basename(file)
        path = os.path.join(res_dir, f'res_{fname}')
        process(file, path, enc=detect_enc(file))


if __name__ == '__main__':
    main(sys.argv[1:])
