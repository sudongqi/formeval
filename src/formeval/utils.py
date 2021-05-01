import json
import collections
import time



def get_wordnet_lemmatizer():
    import nltk
    return nltk.wordnet.WordNetLemmatizer()


def get_regex_tokenizer():
    from nltk.tokenize import RegexpTokenizer
    return RegexpTokenizer('\w+|\$[\d\.]+|\S+')


def load_jsonl(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            json.loads(line)
    return res


def jsonl_iter(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line)


def jsonl_to_dict(data):
    res = collections.defaultdict(list)
    for d in data:
        res[d['id']].append(d['sentence'])
    return res


def load_json(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res

