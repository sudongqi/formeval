import math
import sys
import collections
from .processor import FormProcessor
from ._evaluator import BaseEvaluator
from .ngram import get_ngram_counts


def modified_precision(candidate, references):
    matched_ngram = 0
    total = 0
    for ngram, value in candidate.items():
        total += value
        matched_ngram += min(value, max(r.get(ngram, 0) for r in references))
    return sys.float_info.min if matched_ngram == 0 else matched_ngram / total


def sentences_length(data):
    res = collections.defaultdict(list)
    for key, sentences in data.items():
        for ngram_count in sentences:
            res[key].append(sum(ngram_count.values()))
    return res


def brevity_penalty(candidate_length, reference_lengths):
    best_match = sorted([(abs(r - candidate_length), r) for r in reference_lengths])[0][1]
    return 1 if candidate_length > best_match else math.exp(1 - (best_match / candidate_length))


class _BleuEvaluator(BaseEvaluator):

    def __init__(self, references, processor=None, already_processed=False, silent=True,
                 weights=(0.25, 0.25, 0.25, 0.25)):
        super(_BleuEvaluator, self).__init__(references=references,
                                             processor=processor if processor else FormProcessor(),
                                             already_processed=already_processed,
                                             silent=silent
                                             )
        self.n = len(weights)
        self._references = get_ngram_counts(self._process(references), self.n)
        self.reference_lengths = sentences_length(self._references[0])
        self.weights = weights

    def evaluate(self, candidates):
        _candidates = get_ngram_counts(self._process(candidates), self.n)
        candidate_lengths = sentences_length(_candidates[0])

        scores = collections.defaultdict(list)
        for key in _candidates[0]:
            for j in range(len(_candidates[0][key])):
                score = 0
                print(candidates[key][j])
                for i in range(len(_candidates)):
                    mp = modified_precision(_candidates[i][key][j], self._references[i][key])
                    print(mp)
                    score += self.weights[i] * math.log(mp)
                scores[key].append(
                    brevity_penalty(candidate_lengths[key][j], self.reference_lengths[key]) * math.exp(score))
                print('----------------------------')

        return self.aggregate_scores(scores), scores

    def get_name(self, detailed=True):
        res = 'bleu'
        if detailed:
            res += ' (weights=[{}])'.format(' '.join(self.weights))
        return res
