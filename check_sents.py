import json
import urllib
import string
from functools import partial
from collections import defaultdict, Counter
import os


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


def csv_to_dict(csv_path):
    res = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key, val = line.split(';')
        res[key] = int(val)
    return res


def check_sent(sent, possible_deps):
    """
    Args:
        sent (str): Sentence to check
        possible_deps (set): The set of all possible dependency relations
    """
    test_parsed = mparse(sent)
    test_deps, samples = extract_dependency_features([test_parsed])
    res = 0
    for key in test_deps.keys():
        if key not in possible_deps:
            res += 1
    return res


TH = 0


def get_dep_set(deps):
    possible_deps = set()
    for k, v in deps.items():
        if v > TH:
            possible_deps.add(k)
    return possible_deps


deps = set()


check = partial(check_sent, possible_deps=deps)


def set_deps(deps_path):
    global deps
    deps = get_dep_set(csv_to_dict(deps_path))


def generate_all_sentences(sentence, word_candidates, cur_word, res):
    if cur_word >= len(word_candidates):
        res.append(sentence.copy())
        return
    if word_candidates[cur_word]:
        for c in word_candidates[cur_word][:2]:
            sentence[cur_word] = c
            generate_all_sentences(sentence, word_candidates, cur_word+1, res)
    else:
        generate_all_sentences(sentence, word_candidates, cur_word+1, res)


def get_best_candidate(sentence, word_candidates):
    sents = []
    generate_all_sentences(sentence, word_candidates, 0, sents)
    scores = sorted([(check(' '.join(s)), s) for s in sents], key=lambda x: x[0])
    #  print(scores)
    return scores[0][1]


def fix_sentence(sentence):
    split = nltk.tokenize.word_tokenize(sentence)
    word_candidates = [[word] if word in string.punctuation else corrector.GetCandidates(split, ind) for ind, word in enumerate(split)]
    best = get_best_candidate(split.copy(), word_candidates)
    res = ''
    #  print(word_candidates)
    for ind, w in enumerate(best):
        if w not in string.punctuation:
            res += ' '
        if split[ind][0].isupper():
            res += w.capitalize()
        else:
            res += w
    return res[1:]
