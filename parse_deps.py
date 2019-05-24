import json
import urllib
import os
from glob import glob
import sys
from collections import defaultdict, Counter
import argparse
import re
from functools import partial

import requests as rq
from tqdm import tqdm


proxies = {
  "http": None,
  "https": None,
}


def malt_parse(text, url='http://localhost:2000'):
    rq_url = f'{url}/parse?text={urllib.parse.quote(text)}'
    try:
        res = json.loads(
            rq.get(rq_url, proxies=proxies).content)
    except Exception as err:
        print(f'Fail: {text}, {err}')
        return {}
    r = {}
    for it in res:
        if it['FEATS'] == 'SENT':
            break
        r[it['ID']] = (it['FORM'], it['FEATS'], it['HEAD'], it['DEPREL'])
    return r


if os.environ.get('dockerized', ''):
    mparse = partial(malt_parse, url='http://syntax:2000')
else:
    mparse = malt_parse


def csv_to_dict(csv_path):
    res = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key, val = line.split(';')
        res[key] = int(val)
    return res


false_deps_dict = {}


def extract_dependency_features(data):
    res = Counter()
    samples = defaultdict(lambda: [])
    for sent in tqdm(data):
        for key, val in sent.items():
            # 2 stands for head id
            head = sent.get(val[2], None)
            if not head:
                continue
            rule = str((head[1], val[1], val[3]))
            # if we can't find the feats - quit it
            if head[1] == '-' or val[1] == '-':
                continue
            if rule in false_deps_dict:
                continue
            res[rule] += 1
            samples[rule].append(f'{head[0]} {val[0]}')
    return res, samples


def process(fname):
    # грузим все предложения
    with open(fname, encoding='utf-8') as f:
        sents = f.readlines()
    print('Parsing sentences')
    parsed = [mparse(sents[0])]
    for i in tqdm(range(1, len(sents))):
        parsed.append(mparse(sents[i]))
    print('Extracting deprels')
    return extract_dependency_features(parsed)


def dict_to_csv(csv_path, d):
    with open(csv_path, 'w') as f:
        for k, v in d.items():
            f.write(f'{k};{v}\n')


def process_many(target_dir, res_dir, samples_dir):
    for corp in tqdm(glob(target_dir)):
        r, samples = process(corp)
        fname = os.path.basename(corp)
        dict_to_csv(os.path.join(res_dir, fname), r)
        dict_to_csv(os.path.join(samples_dir, fname), samples)


def samp_to_dict(path):
    res = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key, val = line.split(';')
        res[key] = re.findall(r'\'([\w\s]+)\'', val)
    return res


def combine_processed(proc_dir, samp_dir, res_name='all_res.csv',
                      samp_name='all_samp.csv'):
    rules = Counter()
    print('Combining deps')
    for corp in tqdm(glob(proc_dir)):
        rules.update(csv_to_dict(corp))
    dict_to_csv(res_name, rules)

    print('Combining samples')
    samples = defaultdict(lambda: [])
    for samp in tqdm(glob(samp_dir)):
        samp_dict = samp_to_dict(samp)
        for k, v in samp_dict.items():
            samples[k] += v
    dict_to_csv(samp_name, samples)


def main(args):
    parser = argparse.ArgumentParser(description='Extract dependency relations'
                                                 'from preprocessed corpora.')
    parser.add_argument('--target_dir', help='Target directory containing'
                                             'preprocessed corpora files',
                        required=True)
    parser.add_argument('--res_deps', help='Directory to save syntax parsing'
                                           'results', required=True)
    parser.add_argument('--res_samples', help='Directory to save syntax'
                                              'samples', required=True)
    parser.add_argument('--res_final', help='Name of final result file',
                        default='all_res.csv')
    parser.add_argument('--samp_final', help='Name of final samples file',
                        default='all_samp.csv')
    parser.add_argument('--false_deps', help='Name of final samples file')
    args = parser.parse_args(args)
    target_dir = args.target_dir
    res_deps = args.res_deps
    res_samples = args.res_samples
    res_final = args.res_final
    samp_final = args.samp_final
    false_deps = args.false_deps
    print('Processing files...')

    if not os.path.isdir(res_deps):
        os.makedirs(res_deps)
    if not os.path.isdir(res_samples):
        os.makedirs(res_samples)

    if false_deps:
        false_deps_dict.update(csv_to_dict(false_deps))

    process_many(os.path.join(target_dir, '*.txt'), res_deps, res_samples)
    combine_processed(os.path.join(res_deps, '*.txt'),
                      os.path.join(res_samples, '*.txt'),
                      res_final, samp_final)


if __name__ == '__main__':
    main(sys.argv[1:])
