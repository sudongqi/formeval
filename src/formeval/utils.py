import json
import time
import random
import collections
import os
from statistics import median
from pathlib import Path


def root_dir():
    return str(Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.absolute())


def dict_to_list(d):
    return [v for values in d.values() for v in values]


def _mean(scores):
    if isinstance(scores, dict):
        scores = dict_to_list(scores)
    return sum(scores) / len(scores)


def _median(scores):
    if isinstance(scores, dict):
        scores = dict_to_list(scores)
    return median(scores)


def _max(scores):
    if isinstance(scores, dict):
        scores = dict_to_list(scores)
    return max(scores)


def _harmonic_mean(scores, min_value=0.01):
    if isinstance(scores, dict):
        scores = dict_to_list(scores)
    return len(scores) / sum([1 / max(s, min_value) for s in scores])


class UnknownCandidatesError(Exception):

    def __init__(self, unknown_key):
        super().__init__('key: {} is not found in references'.format(unknown_key))


def sample_from_references(references, discard_identical=True):
    _candidates = collections.defaultdict(list)
    _references = collections.defaultdict(list)
    skip = 1 if discard_identical else 0
    for k, sentences in references.items():
        idx = random.randint(0, len(sentences) - 1)
        _candidates[k].append(sentences[idx])
        _references[k].extend(sentences[:idx] + sentences[idx + skip:])
    return _candidates, _references


class SimpleTimer:
    def __init__(self):
        self.prev = time.time()
        self.start = self.prev

    def check(self):
        res = time.time() - self.prev
        self.prev = time.time()
        return res

    def total_time(self):
        return time.time() - self.start


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


def jsonl_iter_to_dict(data, key, value, allow_duplicated_keys=True):
    res = collections.defaultdict(list)
    for d in data:
        list_or_str = d[value]
        _key = d[key]
        if not allow_duplicated_keys and _key in res:
            continue
        if isinstance(list_or_str, list):
            res[_key].extend(list_or_str)
        elif isinstance(list_or_str, str):
            res[_key].append(list_or_str)
        else:
            raise ValueError('the type of the value has to be either str or list[str]')
    return res


def jsonl_to_dict(path, key='id', value='sentence', allow_duplicated_keys=True):
    return jsonl_iter_to_dict(jsonl_iter(path), key, value, allow_duplicated_keys)


def load_json(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def write_jsonl(data, path):
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
