import time
import collections
from statistics import median
from pathlib import Path
from mbp import *


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


def sample_references(references, n):
    res = collections.defaultdict(list)
    for k, v in references.items():
        random.shuffle(v)
        res[k].extend(v[:n])
    return res


def split_references(references, discard_identical=True):
    _candidates = collections.defaultdict(list)
    _references = collections.defaultdict(list)
    skip = 1 if discard_identical else 0
    num_lonely_references = 0
    for k, sentences in references.items():
        _skip = skip
        if len(sentences) == 1:
            num_lonely_references += 1
            _skip = 0
        idx = random.randint(0, len(sentences) - 1)
        _candidates[k].append(sentences[idx])
        _references[k].extend(sentences[:idx] + sentences[idx + _skip:])
    if num_lonely_references > 0:
        log('{} keys have only 1 reference, this will greatly inflate the score of self-agreement test'.format(
            num_lonely_references), WARNING)
    return _candidates, _references


class Timer:
    def __init__(self):
        self.prev = time.time()
        self.start = self.prev

    def check(self):
        res = time.time() - self.prev
        self.prev = time.time()
        return res

    def total_time(self):
        return time.time() - self.start


def jsonl_iter(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line)


def jsonl_iter_to_dict(data, key, value):
    res = collections.defaultdict(list)
    for d in data:
        list_or_str = d[value]
        _key = d[key]
        if isinstance(list_or_str, list):
            res[_key].extend(list_or_str)
        elif isinstance(list_or_str, str):
            res[_key].append(list_or_str)
        else:
            raise ValueError('the type of the value has to be either str or list[str]')
    return res


def jsonl_to_dict(path, key='id', value='sentence'):
    return jsonl_iter_to_dict(load_jsonl(path), key, value)


def write_jsonl(data, path):
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
