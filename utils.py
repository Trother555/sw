import json
import urllib
import string
from functools import partial
from collections import defaultdict, Counter
import os
import re

import requests as rq
import nltk
import jamspell

corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('ru_small.bin')


nltk.data.path.append('./nltk_data')


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


def extract_dependency_features(data):
    res = Counter()
    samples = defaultdict(lambda: [])
    for sent in data:
        for key, val in sent.items():
            # 2 stands for head id
            head = sent.get(val[2], None)
            if not head:
                continue
            rule = str((head[1], val[1], val[3]))
            res[rule] += 1
            samples[rule].append(f'{head[0]} {val[0]}')
    return res, samples


def samp_to_dict(path):
    res = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key, val = line.split(';')
        res[key] = re.findall(r'\'([\w\s]+)\'', val)
    return res


def dict_to_csv(csv_path, d):
    with open(csv_path, 'w') as f:
        for k, v in d.items():
            f.write(f'{k};{v}\n')


def csv_to_deps(csv_path):
    res = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key, val = line.split(';')
        res[key] = int(val)
    return res


def get_deps_for_sent(sent):
    test_parsed = mparse(sent)
    return extract_dependency_features([test_parsed])


def remove_dep_rel(deps_file, samps_file, deprel):
    samps = samp_to_dict(deps_file)
    deps = csv_to_deps(deps_file)

    if deprel in deps:
        deps.pop(deprel)

    if deprel in samps:
        samps.pop(deprel)

    deps_path = os.path.dirname(deps_file)
    deps_name = os.path.basename(deps_file)
    samps_path = os.path.dirname(samps_file)
    samps_name = os.path.basename(samps_file)

    dict_to_csv(os.path.join(deps_path, f'rm_{deps_name}'), deps)
    dict_to_csv(os.path.join(samps_path, f'rm_{samps_name}'), deps)
