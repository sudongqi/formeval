import math
import sys
import collections
from .processor import FormProcessor
from .base_evaluator import BaseEvaluator
from .ngram import get_ngram_counts


def modified_precision(candidate, references):
    matched_ngram = 0
    total = 0
    for ngram, value in candidate.items():
        total += value
        matched_ngram += min(value, max(r.get(ngram, 0) for r in references))
    return matched_ngram, total


def sentences_length(data):
    res = collections.defaultdict(list)
    for key, sentences in data.items():
        for ngram_count in sentences:
            res[key].append(sum(ngram_count.values()))
    return res


def best_length_match_pair(candidate_length, reference_lengths):
    best_match = sorted([(abs(r - candidate_length), r) for r in reference_lengths])[0][1]
    return candidate_length, best_match


class BleuEvaluator(BaseEvaluator):

    def __init__(self, references, processor=None, already_processed=False, silent=True,
                 weights=(0.25, 0.25, 0.25, 0.25)):
        super(BleuEvaluator, self).__init__(references=references,
                                            processor=processor if processor else FormProcessor(),
                                            already_processed=already_processed,
                                            silent=silent,
                                            has_instance_score=False
                                            )
        self.n = len(weights)
        self._references = get_ngram_counts(self._process(references), self.n)
        self.reference_lengths = sentences_length(self._references[0])
        self.weights = weights
        self.log_setup_finish()

    def aggregate_scores(self, scores):
        res = 0
        for i in range(self.n):
            nominator_sum = 0
            denominator_sum = 0
            for key, data_list in scores.items():
                for data in data_list:
                    nominator, denominator = data[i]
                    nominator_sum += nominator
                    denominator_sum += denominator
            res += self.weights[i] * math.log(max(sys.float_info.min, nominator_sum) / max(denominator_sum, 1))
        c = 0
        r = 0
        for key, data_list in scores.items():
            for data in data_list:
                candidate_length, best_match_length = data[self.n]
                c += candidate_length
                r += best_match_length
        bp = 1 if c > r else math.exp(1 - (r / c))
        return bp * math.exp(res)

    def evaluate(self, candidates):
        self.log_evaluation_start(candidates)
        _candidates = get_ngram_counts(self._process(candidates), self.n)
        candidate_lengths = sentences_length(_candidates[0])

        scores = collections.defaultdict(list)
        for key in _candidates[0]:
            for j in range(len(_candidates[0][key])):
                candidate_score = 0
                scores[key].append([])
                for i in range(len(_candidates)):
                    numerator, denominator = modified_precision(_candidates[i][key][j], self._references[i][key])
                    scores[key][-1].append((numerator, denominator))
                    candidate_score += self.weights[i] * math.log(
                        max(sys.float_info.min, numerator) / max(1, denominator))
                _c, _r = best_length_match_pair(candidate_lengths[key][j], self.reference_lengths[key])
                candidate_score = (1 if _c > _r else math.exp(1 - (_r / _c))) * math.exp(candidate_score)
                scores[key][-1].append((_c, _r))
                scores[key][-1].append(candidate_score)
        _score = self.aggregate_scores(scores)
        _scores = {key: [score[-1] for score in score_list] for key, score_list in scores.items()}
        self.log_evaluation_finish()
        return _score, _scores

    def get_name(self, detailed=True):
        res = 'bleu'
        if detailed:
            res += ' (weights=[{}])'.format(' '.join(map(str, self.weights)))
        return res
